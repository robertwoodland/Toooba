import Types::*;
import ProcTypes::*;
import RegFile::*;
import Vector::*;
import BrPred::*;
import GlobalBrHistReg::*;
import Ehr::*;
import Real :: * ;

export PerceptronTrainInfo(..);
export mkPerceptron;
export PerceptronEntries;
export PerceptronGHistEntries;
export PerceptronGHist;
export PerceptronIndex;
export PerceptronIndexWidth;
export PerceptronCount;
export PerceptronsRegIndex;
export PerceptronsRegIndexWidth;
export AddrRange;
export AddrWidth;

// Local Perceptron Typedefs
typedef 63 PerceptronEntries; // Numeric: Size of perceptron (length of history and weights) - typically 4 to 66 depending on hardware budget.
typedef TLog#(TAdd#(PerceptronEntries, 1)) PerceptronIndexWidth; // Numeric: Number of bits to be used for indexing history and weights. 1 is to ensure index big enough to deal with biases.
typedef Bit#(PerceptronIndexWidth) PerceptronIndex; // Value: Bits used as the index for history and weights.

// TODO (RW): Allow size of global history to be different to that of each local history
typedef PerceptronEntries PerceptronGHistEntries; // Numeric: Size of global history
typedef Bit#(PerceptronGHistEntries) PerceptronGHist; // Value: Bits used as the global history.
typedef GlobalBrHistReg#(PerceptronGHistEntries) PerceptronGHistReg; // Register: Global history register.

typedef SizeOf#(Addr) AddrWidth; // Numeric: Number of bits in an address.
typedef TExp#(AddrWidth) AddrRange; // Numeric: Number of addresses in the range.
// typedef TDiv#(AddrRange, TExp#(40)) PerceptronCount; // Numeric: Number of perceptrons - depends on hash function. Made smaller as would take ages to initialise...
typedef 16 PerceptronCount; // Numeric: Number of perceptrons - depends on hash function. Made smaller as would take ages to initialise...
// TODO (RW): Make this same size as BHT. Look at papers to see what is a reasonable size.
typedef TLog#(PerceptronCount) PerceptronsRegIndexWidth; // Numeric: Number of bits to be used for indexing the Regfile of perceptrons.
typedef Bit#(PerceptronsRegIndexWidth) PerceptronsRegIndex; // Value: Bits used as the index for the Regfile.
 
// bookkeeping info a branch should keep for future training
typedef struct {
    PerceptronGHist gHist;
    PerceptronsRegIndex index;
} PerceptronTrainInfo deriving(Bits, Eq, FShow);

typedef Vector#(PerceptronEntries, Bool) PerceptronHistory;
typedef Vector#(TAdd#(PerceptronEntries, 1), Int#(8)) PerceptronWeights;
typedef Vector#(PerceptronGHistEntries, Int#(8)) PerceptronGWeights;

interface PerceptronHistorian; // Not stateful
    method PerceptronHistory update(PerceptronHistory hist, Bool taken);
    method Bool get(PerceptronHistory hist, PerceptronIndex index); // TODO (RW): What happens if you call with a value bigger than PerceptronEntries?
    method PerceptronHistory initHist();
endinterface

module mkPerceptronHistorianShift(PerceptronHistorian);
    // TODO (RW): Could define another implementation which uses a head pointer and overwrites oldest value on update.

    method PerceptronHistory update(PerceptronHistory hist, Bool taken);
        // shift all history values down one, add new value at the top.
        for (PerceptronIndex i = fromInteger(valueOf(PerceptronEntries)) - 1; i > 0; i = i - 1) begin
            hist[i] = hist[i - 1];
        end
        hist[0] = taken;
        return hist; // Can't update history in place as it can't be a reg.
    endmethod

    method Bool get(PerceptronHistory hist, PerceptronIndex index);
        return hist[index];
    endmethod

    method PerceptronHistory initHist;
        PerceptronHistory hist = replicate(False);
        return hist;
    endmethod
endmodule

(* synthesize *)
module mkPerceptron(DirPredictor#(PerceptronTrainInfo));
    PerceptronHistorian ph <- mkPerceptronHistorianShift;
    RegFile#(PerceptronsRegIndex, PerceptronHistory) histories <- mkRegFileWCF(0,fromInteger(valueOf(PerceptronCount)-1));
    PerceptronGHistReg global_history <- mkGlobalBrHistReg;
    RegFile#(PerceptronsRegIndex, PerceptronWeights) weights <- mkRegFileWCF(0,fromInteger(valueOf(PerceptronCount)-1)); 
    RegFile#(PerceptronsRegIndex, PerceptronGWeights) global_weights <- mkRegFileWCF(0,fromInteger(valueOf(PerceptronCount)-1)); 
    // TODO (RW): Decide max weight size and prevent overflow. 8 suggested in paper.
    
    Reg#(Addr) pc_reg <- mkRegU;
    Reg#(Int#(16)) trainCount <- mkReg(0); // TODO (RW): Choose a proper type for this that can't be too small for PerceptronEntries
    
    // EHR to record predict results in this cycle
    Ehr#(TAdd#(1, SupSize), Bit#(TLog#(TAdd#(SupSize, 1)))) predCnt <- mkEhr(0);
    Ehr#(TAdd#(1, SupSize), Bit#(SupSize)) predRes <- mkEhr(0);

    Reg#(PerceptronsRegIndex) nextInit <- mkReg(0);
    Reg#(Bool) resetHist <- mkReg(True);
    PerceptronWeights zeroWeights = replicate(0);
    PerceptronGWeights zeroGWeights = replicate(0);
        
    rule initHistory(resetHist);
        if (nextInit <= fromInteger(valueOf(PerceptronCount) - 1)) begin
            histories.upd(nextInit, ph.initHist());
            weights.upd(nextInit, zeroWeights); // TODO (RW): Consider what happens at start when history is full of Falses.
            global_weights.upd(nextInit, zeroGWeights);
        end
        if (nextInit == fromInteger(valueOf(PerceptronCount) - 1)) begin
            // $display("BSV Perceptron Init: Initialised all perceptrons & hists");
            resetHist <= False;
        end

        nextInit <= (nextInit == fromInteger(valueOf(PerceptronCount) - 1)) ? 0 : nextInit + 1;

        // TODO (RW): Should global (history) be done in a separate rule? - just initialise when made. Is it even done atm?
    endrule

    function PerceptronsRegIndex getIndex(Addr pc); // TODO (RW): Try better hash functions?
        return truncate(pc >> 1); // compressed instructions
    endfunction

    // Function to compute the perceptron output
    function Bool computePerceptronOutput(PerceptronWeights weight, PerceptronHistory history, PerceptronGWeights glob_weight, PerceptronGHistReg global_hist); // TODO (RW): Can make actionvalue for debug prints. Set back after for performance.
        let gHist = global_hist.history; // Bit#(...)

        Int#(16) sum = extend(weight[0]); // Bias
        for (Integer i = 1; i <= valueOf(PerceptronEntries); i = i + 1) begin // TODO (RW): check loop boundary
            sum = boundedPlus(sum, (history[i-1] ? extend(weight[i]) : extend(-weight[i]))); // Think about hardware this implies. - log (128) = 9 deep?
        end
        for (Integer i = 0; i < valueOf(PerceptronGHistEntries); i = i + 1) begin // TODO (RW): check loop boundary
            // TODO (RW): Don't need to misalign for ghist as not using a global bias
            sum = boundedPlus(sum, ((gHist[i] == 1) ? extend(glob_weight[i]) : extend(-glob_weight[i])));
        end
        return sum >= 0;
    endfunction

    PerceptronGHist curGHist = global_history.history; // global history: MSB is the latest branch

    // Interface for each perceptron in the table - is this true?
    // What is SupSize? Seems to be the number of perceptrons I actually can use?
    Vector#(SupSize, DirPred#(PerceptronTrainInfo)) predIfc;
    for(Integer i = 0; i < valueOf(SupSize); i = i+1) begin
        predIfc[i] = (interface DirPred;
            method ActionValue#(DirPredResult#(PerceptronTrainInfo)) pred() if (!resetHist) = actionvalue // Guarded on resetHist
                // get the global history
                // all previous branch in this cycle must be not taken
                // otherwise this branch should be on wrong path
                // because all inst in same cycle are fetched consecutively
                PerceptronGHist gHist = curGHist >> predCnt[i];
                
                // Don't need to do?
                // let index = getIndex(offsetPc(pc_reg, i), gHist);

                let index = getIndex(offsetPc(pc_reg, i));

                // In pred, most recent is correct
                PerceptronGHist globHist = global_history.history;                

                Bool taken = computePerceptronOutput(weights.sub(index), histories.sub(index), global_weights.sub(index), global_history); // TODO (RW): Work out how to pass
                // TODO (RW): Need to know how to flush global_history on mispred? Check other predictors that use global (GSelect).

                // $display("BSV Perceptron Pred %d: Taken: %d", index, taken);

                // record pred result (for global history)
                predCnt[i] <= predCnt[i] + 1;
                Bit#(SupSize) res = predRes[i];
                res[predCnt[i]] = pack(taken);
                predRes[i] <= res;

                return DirPredResult {
                    taken: taken,
                    train: PerceptronTrainInfo {
                        gHist: globHist,
                        index: index
                    }
                };
            endactionvalue;
        endinterface);
    end

    (* fire_when_enabled, no_implicit_conditions *)
    rule canonGlobalHist;
        global_history.addHistory(predRes[valueof(SupSize)], predCnt[valueof(SupSize)]);
        predRes[valueof(SupSize)] <= 0;
        predCnt[valueof(SupSize)] <= 0;
    endrule

    method nextPc = pc_reg._write;

    interface pred = predIfc;

    
    
    method Action update(Bool taken, PerceptronTrainInfo train, Bool mispred) if (!resetHist); 
        let index = train.index; // already hashed
        
        // update history if mispred
        if (mispred) begin
            PerceptronGHist newHist = truncate({pack(taken), train.gHist} >> 1);
            global_history.redirect(newHist);
        end 
    
        // TODO (RW): Only train if below training threshold. Paper says threshold = 1.93 * branch history + 14. This could be a power optimisation. Test with and without, measure impact.
        
        
        let local_hist = histories.sub(index);
        PerceptronWeights local_weights = weights.sub(index);
        PerceptronGWeights g_weights = global_weights.sub(index);
        
        // Train bias
        local_weights[0] = boundedPlus(local_weights[0], ((taken) ? 1 : -1));
        // TODO (RW): Why isn't this updating (sits at 0)

        // Train local and global weights
        
        // Bool localCorrelationPos, globCorrelationPos;
        // Int#(8) localInc, globInc;
        if (mispred || (trainCount < fromInteger(trunc(1.93 * (fromInteger(valueOf(PerceptronEntries))) + 14)))) begin
            // $display("BSV Perceptron Update: Training count %d, Mispred? %b", trainCount, mispred);
            // $display("BSV Perceptron Update: Local Hist %d: %b", index, local_hist);
            for (Integer i = 1; i <= valueOf(PerceptronEntries); i = i + 1) begin 
                // Paper's update
                local_weights[i] = boundedPlus(local_weights[i], ((local_hist[i-1] == taken) ? 1 : -1));
                
                // // Penalise incorrect weights by subtracting 10 instead of 1.
                // if (local_weights[i] != 0) begin
                //     localCorrelationPos = ((local_hist[i-1] ? 1 : -1) * local_weights[i]) > 0;
                //     localInc = (local_weights[i] > 0) ? 1 : -1;
                //     local_weights[i] = boundedPlus(local_weights[i], localInc * ((localCorrelationPos == taken) ? 1 : -10));
                // end else begin
                //     local_weights[i] = (local_hist[i-1] == taken) ? 1 : -1;
                // end
                
                // $display("BSV Perceptron Update Local Weights %d Post Update %d: %d", index, i, local_weights[i]); 
            end
            
            for (Integer i = 0; i < valueOf(PerceptronGHistEntries); i = i + 1) begin
                g_weights[i] = boundedPlus(g_weights[i], (((train.gHist[i] != 0) == taken) ? 1 : -1)); 
                // // Penalise incorrect weights by subtracting 10 instead of 1.
                // if (g_weights[i] != 0) begin
                //     globCorrelationPos = (((train.gHist[i-1] != 0) ? 1 : -1) * g_weights[i]) > 0;
                //     globInc = (g_weights[i] > 0) ? 1 : -1;
                //     g_weights[i] = boundedPlus(g_weights[i], globInc * ((globCorrelationPos == taken) ? 1 : -10));
                // end else begin
                //     g_weights[i] = ((train.gHist[i-1] != 0) == taken) ? 1 : -1;
                // end
                // $display("BSV Perceptron Update Global Weights Post Update %d: %d", i, g_weights[i]); 
            end

        
            // Update weights!
            global_weights.upd(index, g_weights);
            trainCount <= boundedPlus(trainCount, 1);

        end // (training)
        
        weights.upd(index, local_weights);
        
        // Update local history
        local_hist = ph.update(local_hist, taken);
        // $display("BSV Global Weights Post Update %d: %b", index, g_weights);
        // $display("BSV Perceptron Update: Global Hist Pre Update: %b", train.gHist);
        // $display("BSV Perceptron Update: Local Hist %d Post Update: %b", index, local_hist);
        // $display("BSV Perceptron Update: Local Weights %d: %b", index, local_weights);

        histories.upd(index, local_hist);
    endmethod


    // Perceptron predictor also doesn't need to be flushed
    method flush = noAction;
    method flush_done = True;
endmodule

