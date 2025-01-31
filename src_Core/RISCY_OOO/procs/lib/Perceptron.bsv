import Types::*;
import ProcTypes::*;
import RegFile::*;
import Vector::*;
import BrPred::*;
import GlobalBrHistReg::*;
import Ehr::*;

export PerceptronTrainInfo;
export mkPerceptron;
export PerceptronEntries;
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
typedef Bit#(PerceptronEntries) PerceptronGHist; // Value: Bits used as the global history.
typedef GlobalBrHistReg#(PerceptronEntries) PerceptronGHistReg; // Register: Global history register.
typedef TLog#(TAdd#(PerceptronEntries, 1)) PerceptronIndexWidth; // Numeric: Number of bits to be used for indexing history and weights. 1 is to ensure index big enough to deal with biases.
typedef Bit#(PerceptronIndexWidth) PerceptronIndex; // Value: Bits used as the index for history and weights.

typedef SizeOf#(Addr) AddrWidth; // Numeric: Number of bits in an address.
typedef TExp#(AddrWidth) AddrRange; // Numeric: Number of addresses in the range.
typedef TDiv#(AddrRange, 4) PerceptronCount; // Numeric: Number of perceptrons - depends on hash function.
typedef TLog#(PerceptronCount) PerceptronsRegIndexWidth; // Numeric: Number of bits to be used for indexing the Regfile of perceptrons.
typedef Bit#(PerceptronsRegIndexWidth) PerceptronsRegIndex; // Value: Bits used as the index for the Regfile.
 
// bookkeeping info a branch should keep for future training
typedef struct {
    PerceptronGHist gHist;
    PerceptronsRegIndex index;
} PerceptronTrainInfo deriving(Bits, Eq, FShow);

typedef Vector#(PerceptronEntries, Bool) PerceptronHistory;
typedef Vector#(TAdd#(PerceptronEntries, 1), Int#(8)) PerceptronWeights;

interface PerceptronHistorian; // Not stateful
    method PerceptronHistory update(PerceptronHistory hist, Bool taken);
    method Bool get(PerceptronHistory hist, PerceptronIndex index); // TODO (RW): What happens if you call with a value bigger than PerceptronEntries?
    method PerceptronHistory initHist();
endinterface

module mkPerceptronHistorianShift(PerceptronHistorian);
    // TODO (RW): Could define another implementation which uses a head pointer and overwrites oldest value on update.

    method PerceptronHistory update(PerceptronHistory hist, Bool taken);
        // shift all history values down one, add new value at the top.
        for (PerceptronIndex i = 1; i < fromInteger(valueOf(PerceptronEntries)); i = i + 1) begin
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
    RegFile#(PerceptronsRegIndex, PerceptronWeights) global_weights <- mkRegFileWCF(0,fromInteger(valueOf(PerceptronCount)-1)); 
    
    Reg#(Addr) pc_reg <- mkRegU;
    // TODO (RW): Decide max weight size and prevent overflow. 8 suggested in paper.
    // TODO (RW): Allow size of global history to be different to that of each local history
    
    // EHR to record predict results in this cycle
    Ehr#(TAdd#(1, SupSize), Bit#(TLog#(TAdd#(SupSize, 1)))) predCnt <- mkEhr(0);
    Ehr#(TAdd#(1, SupSize), Bit#(SupSize)) predRes <- mkEhr(0);

    Reg#(PerceptronsRegIndex) i <- mkReg(0);
    Reg#(Bool) resetHist <- mkReg(True);
        
    rule initHistory(resetHist);
        if (i <= fromInteger(valueOf(PerceptronCount) - 1)) begin
            histories.upd(i, ph.initHist());
            weights.upd(i, replicate(0)); // TODO (RW): Consider what happens at start when history is full of Falses.
            global_weights.upd(i, replicate(0));
        end
        if (i == fromInteger(valueOf(PerceptronCount) - 1)) begin
            resetHist <= False;
        end

        i <= (i == fromInteger(valueOf(PerceptronCount) - 1)) ? 0 : i + 1;

        // TODO (RW): Should global be done in a separate rule? - just initialise when made
        // TODO (RW): May need to guard things on not resetHist -> method stuff on history can only be done if not resetHist.
    endrule

    function PerceptronsRegIndex getIndex(Addr pc);
        return truncate(pc >> 2);
    endfunction

    // function PerceptronsRegIndex getIndex(Addr pc, PerceptronGHist gHist);
    //     return {gHist, truncate(pc >> 2)};
    // endfunction

    // Function to compute the perceptron output
    function Bool computePerceptronOutput(PerceptronWeights weight, PerceptronHistory history, PerceptronWeights glob_weight, PerceptronGHistReg global_hist);
        let gHist = global_hist.history; // Bit#(...)

        Int#(16) sum = extend(weight[0]); // Bias weight - TODO (RW): check this can't overflow.
        for (Integer i = 1; i < valueOf(PerceptronEntries); i = i + 1) begin // TODO (RW): check loop boundary
            sum = sum + (history[i] ? extend(weight[i]) : extend(-weight[i])); // Think about hardware this implies. - log (128) = 9 deep?
            // TODO (RW): Add parameter to choose how much to use global history (multiplier)
            sum = sum + ((gHist[i] == 1) ? extend(glob_weight[i]) : extend(-glob_weight[i]));
        end
        return sum >= 0;
    endfunction

    PerceptronGHist curGHist = global_history.history; // global history: MSB is the latest branch

    // Interface for each perceptron in the table
    Vector#(SupSize, DirPred#(PerceptronTrainInfo)) predIfc;
    for(Integer i = 0; i < valueOf(SupSize); i = i+1) begin
        predIfc[i] = (interface DirPred;
            method ActionValue#(DirPredResult#(PerceptronTrainInfo)) pred = actionvalue
                // get the global history
                // all previous branch in this cycle must be not taken
                // otherwise this branch should be on wrong path
                // because all inst in same cycle are fetched consecutively
                PerceptronGHist gHist = curGHist >> predCnt[i];
                
                // Don't need to do?
                // let index = getIndex(offsetPc(pc_reg, i), gHist);

                let index = getIndex(offsetPc(pc_reg, i));
                Bool taken = computePerceptronOutput(weights.sub(index), histories.sub(index), global_weights.sub(index), global_history); // TODO (RW): Work out how to pass
                // TODO (RW): Need to know how to flush global_history on mispred? Check other predictors that use global (GSelect).

                // record pred result (for global history)
                predCnt[i] <= predCnt[i] + 1;
                Bit#(SupSize) res = predRes[i];
                res[predCnt[i]] = pack(taken);
                predRes[i] <= res;

                return DirPredResult {
                    taken: taken,
                    train: PerceptronTrainInfo {
                        gHist: gHist,
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

    
    method Action update(Bool taken, PerceptronTrainInfo train, Bool mispred); 
        // update history if mispred
        if (mispred) begin
            PerceptronGHist newHist = truncate({pack(taken), train.gHist} >> 1);
            global_history.redirect(newHist);
        end
    
        // TODO (RW): Only train if below training threshold. Paper says threshold = 1.93 * branch history + 14. This could be a power optimisation. Test with and without, measure impact.
        
        let index = train.index; // already hashed
        let local_hist = histories.sub(index);
        PerceptronWeights local_weights = weights.sub(index);
        
        // Increment bias if taken, else decrement
        local_weights[0] = (taken) ? local_weights[0] + 1 : local_weights[0] - 1;

        for (Integer i = 1; i < valueOf(PerceptronEntries); i = i + 1) begin
            local_weights[i] = local_weights[i] + (taken == local_hist[i] ? 1 : -1);
        end

        // Update local history
        local_hist = ph.update(local_hist, taken);
        histories.upd(index, local_hist);
    endmethod


    // Perceptron predictor also doesn't need to be flushed
    method flush = noAction;
    method flush_done = True;
endmodule

