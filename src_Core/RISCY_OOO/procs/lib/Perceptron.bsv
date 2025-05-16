import Types::*;
import ProcTypes::*;
import RegFile::*;
import Vector::*;
import BrPred::*;
import GlobalBrHistReg::*;
import Ehr::*;
import Real::* ;

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
typedef 14 PerceptronEntries; // Numeric: Size of perceptron (length of history and weights) - typically 4 to 66 depending on hardware budget.
typedef TLog#(TAdd#(PerceptronEntries, 1)) PerceptronIndexWidth; // Numeric: Number of bits to be used for indexing history and weights. 1 is to ensure index big enough to deal with biases.
typedef Bit#(PerceptronIndexWidth) PerceptronIndex; // Value: Bits used as the index for history and weights.
typedef TAdd#(TMul#(PerceptronEntries, 2), 14) Threshold;

typedef 71 PerceptronGHistEntries; // Numeric: Size of global history
typedef Bit#(PerceptronGHistEntries) PerceptronGHist; // Value: Bits used as the global history.
typedef GlobalBrHistReg#(PerceptronGHistEntries) PerceptronGHistReg; // Register: Global history register.

typedef SizeOf#(Addr) AddrWidth; // Numeric: Number of bits in an address.
typedef TExp#(AddrWidth) AddrRange; // Numeric: Number of addresses in the range.
// typedef TDiv#(AddrRange, TExp#(40)) PerceptronCount; // Numeric: Number of perceptrons - depends on hash function. Made smaller as would take ages to initialise...
typedef 750 PerceptronCount; // Numeric: Number of perceptrons - depends on hash function. Made smaller as would take ages to initialise...
// TODO (RW): Make this same size as BHT. Look at papers to see what is a reasonable size.
typedef TLog#(PerceptronCount) PerceptronsRegIndexWidth; // Numeric: Number of bits to be used for indexing the Regfile of perceptrons.
typedef Bit#(PerceptronsRegIndexWidth) PerceptronsRegIndex; // Value: Bits used as the index for the Regfile.
 
// bookkeeping info a branch should keep for future training
typedef struct {
    PerceptronGHist gHist;
    PerceptronsRegIndex index;
    Bool train;
} PerceptronTrainInfo deriving(Bits, Eq, FShow);

typedef Vector#(PerceptronEntries, Bool) PerceptronHistory;
typedef Vector#(TAdd#(PerceptronEntries, 1), Int#(8)) PerceptronWeights;
typedef Vector#(PerceptronGHistEntries, Int#(8)) PerceptronGWeights;

interface PerceptronHistorian; // Not stateful
    method PerceptronHistory update(PerceptronHistory hist, Bool taken);
    method Bool get(PerceptronHistory hist, PerceptronIndex index); // TODO (RW): What happens if you call with a value bigger than PerceptronEntries?
    method PerceptronHistory initHist();
    // TODO (RW): Rename to reset?
    // TODO (RW): Don't init local & global hist? 101010 may be fairer with an initial history of 000000. Saves time too.
endinterface

module mkPerceptronHistorianShift(PerceptronHistorian);
    // TODO (RW): Could define another implementation which uses a head pointer and overwrites oldest value on update.

    method PerceptronHistory update(PerceptronHistory hist, Bool taken);
        // shift all history values down one, add new value at the top.
        // TODO (RW): Try using rotate method here?
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


interface HashFunction#(type perceptronsRegIndex);
    method perceptronsRegIndex getIndex(Addr pc);
endinterface

module mkTruncate(HashFunction#(PerceptronsRegIndex));
    method PerceptronsRegIndex getIndex(Addr pc);
        return truncate(pc >> 1); // compressed instructions
    endmethod
endmodule

module mkFold(HashFunction#(PerceptronsRegIndex));
    method PerceptronsRegIndex getIndex(Addr pc);
        PerceptronsRegIndex folded = 0;

        // Break PC into chunks of size PerceptronsRegIndexWidth
        for (Integer i = 0; i < valueOf(AddrWidth); i = i + valueOf(PerceptronsRegIndexWidth)) begin
            PerceptronsRegIndex chunk = truncate(pc >> i); // get chunk of appropriate size
            folded = folded ^ chunk;       // XOR fold it in
        end

        return folded;
    endmethod
endmodule

module mkMod(HashFunction#(PerceptronsRegIndex));
    method PerceptronsRegIndex getIndex(Addr pc);
        Addr modded = (pc % fromInteger(valueOf(PerceptronCount)));
        return truncate(modded);
    endmethod
endmodule

module mkFoldDrop(HashFunction#(PerceptronsRegIndex));
    method PerceptronsRegIndex getIndex(Addr pc);
        // Break PC into chunks of size PerceptronsRegIndexWidth
        PerceptronsRegIndex folded = 0;
        for (Integer i = 0; i < valueOf(AddrWidth); i = i + valueOf(PerceptronsRegIndexWidth)) begin
            PerceptronsRegIndex chunk = truncate(pc >> i); // get chunk of appropriate size
            folded = folded ^ chunk;       // XOR fold it in
        end

        // If out of range, drop MSB
        if (folded > fromInteger(valueOf(PerceptronCount) - 1)) begin
            folded = (truncate(folded << 1) >> 1);
        end
        
        // Return the final index
        return folded;
    endmethod
endmodule

module mkFoldMod(HashFunction#(PerceptronsRegIndex));
    method PerceptronsRegIndex getIndex(Addr pc);
        // Break PC into chunks of size PerceptronsRegIndexWidth
        PerceptronsRegIndex folded = 0;
        for (Integer i = 0; i < valueOf(AddrWidth); i = i + valueOf(PerceptronsRegIndexWidth)) begin
            PerceptronsRegIndex chunk = truncate(pc >> i); // get chunk of appropriate size
            folded = folded ^ chunk;       // XOR fold it in
        end

        Bit#(TAdd#(PerceptronsRegIndexWidth, 1)) index = zeroExtend(folded);
        index = index % fromInteger(valueOf(PerceptronCount));

        // Return the final index
        return truncate(index);
    endmethod
endmodule



module mkHybridMod(HashFunction#(PerceptronsRegIndex));
    method PerceptronsRegIndex getIndex(Addr pc);
        PerceptronsRegIndex folded = 0;
        UInt#(TAdd#(PerceptronsRegIndexWidth, 1)) count = fromInteger(valueOf(PerceptronCount));

        // Break PC into chunks of size PerceptronsRegIndexWidth
        for (Integer i = 0; i < valueOf(AddrWidth); i = i + valueOf(PerceptronsRegIndexWidth)) begin
            PerceptronsRegIndex chunk = truncate(pc >> i); // get chunk of appropriate size
            folded = folded ^ chunk;       // XOR fold it in
        end

        // If a power of two, just truncate to size
        if ((count & (count - 1)) != 0) begin
            Bit#(TAdd#(PerceptronsRegIndexWidth, 1)) index = zeroExtend(folded);
            index = index % fromInteger(valueOf(PerceptronCount));

            folded = truncate(index);
        end

        // Return the final index
        return folded;
    endmethod
endmodule


module mkHybridMod1(HashFunction#(PerceptronsRegIndex));
    method PerceptronsRegIndex getIndex(Addr pc);
        Bit#(TAdd#(PerceptronsRegIndexWidth, 1)) folded = 0;
        UInt#(TAdd#(PerceptronsRegIndexWidth, 1)) count = fromInteger(valueOf(PerceptronCount));

        // Break PC into chunks of size PerceptronsRegIndexWidth
        for (Integer i = 0; i < valueOf(AddrWidth); i = i + valueOf(PerceptronsRegIndexWidth) + 1) begin
            Bit#(TAdd#(PerceptronsRegIndexWidth, 1)) chunk = truncate(pc >> i); // get chunk of appropriate size
            folded = folded ^ chunk;       // XOR fold it in
        end

        PerceptronsRegIndex index;
        // If a power of two, just truncate to size
        if ((count & (count - 1)) == 0) begin
            index = truncate(folded);
        end else begin
            // Try doing the expensive thing... MOD(valueOf(PerceptronCount))
            folded = folded % fromInteger(valueOf(PerceptronCount));
            index = truncate(folded);
        end

        // Return the final index
        return index;
    endmethod
endmodule

module mkHybridMod2(HashFunction#(PerceptronsRegIndex));
    method PerceptronsRegIndex getIndex(Addr pc);
        Bit#(TAdd#(PerceptronsRegIndexWidth, 2)) folded = 0;
        UInt#(TAdd#(PerceptronsRegIndexWidth, 1)) count = fromInteger(valueOf(PerceptronCount));

        // Break PC into chunks of size PerceptronsRegIndexWidth
        for (Integer i = 0; i < valueOf(AddrWidth); i = i + valueOf(PerceptronsRegIndexWidth) + 2) begin
            Bit#(TAdd#(PerceptronsRegIndexWidth, 2)) chunk = truncate(pc >> i); // get chunk of appropriate size
            folded = folded ^ chunk;       // XOR fold it in
        end

        PerceptronsRegIndex index;
        // If a power of two, just truncate to size
        if ((count & (count - 1)) == 0) begin
            index = truncate(folded);
        end else begin
            folded = folded % fromInteger(valueOf(PerceptronCount));
            index = truncate(folded);
        end

        // Return the final index
        return index;
    endmethod
endmodule

module mkHybridMod3(HashFunction#(PerceptronsRegIndex));
    method PerceptronsRegIndex getIndex(Addr pc);
        Bit#(TAdd#(PerceptronsRegIndexWidth, 3)) folded = 0;
        UInt#(TAdd#(PerceptronsRegIndexWidth, 1)) count = fromInteger(valueOf(PerceptronCount));

        // Break PC into chunks of size PerceptronsRegIndexWidth
        for (Integer i = 0; i < valueOf(AddrWidth); i = i + valueOf(PerceptronsRegIndexWidth) + 3) begin
            Bit#(TAdd#(PerceptronsRegIndexWidth, 3)) chunk = truncate(pc >> i); // get chunk of appropriate size
            folded = folded ^ chunk;       // XOR fold it in
        end

        PerceptronsRegIndex index;
        // If a power of two, just truncate to size
        if ((count & (count - 1)) == 0) begin
            index = truncate(folded);
        end else begin
            folded = folded % fromInteger(valueOf(PerceptronCount));
            index = truncate(folded);
        end

        // Return the final index
        return index;
    endmethod
endmodule

module mkHybridDrop(HashFunction#(PerceptronsRegIndex));
    method PerceptronsRegIndex getIndex(Addr pc);
        PerceptronsRegIndex folded = 0;
        UInt#(TAdd#(PerceptronsRegIndexWidth, 1)) count = fromInteger(valueOf(PerceptronCount));
        
        // Break PC into chunks of size PerceptronsRegIndexWidth
        for (Integer i = 0; i < valueOf(AddrWidth); i = i + valueOf(PerceptronsRegIndexWidth)) begin
            PerceptronsRegIndex chunk = truncate(pc >> i); // get chunk of appropriate size
            folded = folded ^ chunk;       // XOR fold it in
        end

        // If a power of two, just return
        // Otherwise:
        if ((count & (count - 1)) != 0) begin
            // If out of range, drop MSB
            if (folded > fromInteger(valueOf(PerceptronCount) - 1)) begin
                folded = (truncate(folded << 1) >> 1);
            end
        end

        // Return the final index
        return folded;
    endmethod
endmodule


(* synthesize *)
module mkPerceptron(DirPredictor#(PerceptronTrainInfo));
    HashFunction#(PerceptronsRegIndex) hash <- mkHybridMod3;
    PerceptronHistorian ph <- mkPerceptronHistorianShift;
    RegFile#(PerceptronsRegIndex, PerceptronHistory) histories <- mkRegFileWCF(0,fromInteger(valueOf(PerceptronCount)-1));
    PerceptronGHistReg global_history <- mkGlobalBrHistReg;
    RegFile#(PerceptronsRegIndex, PerceptronWeights) weights <- mkRegFileWCF(0,fromInteger(valueOf(PerceptronCount)-1)); 
    RegFile#(PerceptronsRegIndex, PerceptronGWeights) global_weights <- mkRegFileWCF(0,fromInteger(valueOf(PerceptronCount)-1)); 
    
    Reg#(Addr) pc_reg <- mkRegU;
    // TODO (RW): Decide max weight size and prevent overflow. 8 suggested in paper.
    
    // EHR to record predict results in this cycle
    Ehr#(TAdd#(1, SupSize), Bit#(TLog#(TAdd#(SupSize, 1)))) predCnt <- mkEhr(0);
    Ehr#(TAdd#(1, SupSize), Bit#(SupSize)) predRes <- mkEhr(0);

    Reg#(PerceptronsRegIndex) nextInit <- mkReg(0);
    Reg#(Bool) resetHist <- mkReg(True);
    PerceptronWeights zeroWeights = replicate(0);
    PerceptronGWeights zeroGWeights = replicate(0);
        
    rule initHistory(resetHist);
        if (nextInit <= fromInteger(valueOf(PerceptronCount) - 1)) begin
            // $display("BSV Perceptron Init: Uninitialised local history: %b", histories.sub(nextInit));
            // let weight = weights.sub(nextInit)[0];
            // $display("BSV Perceptron Init: Uninitialised local weights: %d", weight);
            histories.upd(nextInit, ph.initHist());
            weights.upd(nextInit, zeroWeights); // TODO (RW): Consider what happens at start when history is full of Falses.
            global_weights.upd(nextInit, zeroGWeights);
        end
        if (nextInit == fromInteger(valueOf(PerceptronCount) - 1)) begin
            // $display("BSV Perceptron Init: Initialised all perceptrons & hists");
            resetHist <= False;
        end

        nextInit <= (nextInit == fromInteger(valueOf(PerceptronCount) - 1)) ? 0 : nextInit + 1;
    endrule

    function PerceptronsRegIndex getIndex(Addr pc);
        return hash.getIndex(pc);
    endfunction

    // Function to compute the perceptron output
    function Int#(16) computePerceptronOutput(PerceptronWeights weight, PerceptronHistory history, PerceptronGWeights glob_weight, PerceptronGHistReg global_hist);
        let gHist = global_hist.history; // Bit#(...)

        // TODO (RW): Dynamically choose a type based on the size of the weights, and so the max value
        Int#(16) sum = extend(weight[0]); // Bias
        for (Integer i = 1; i <= valueOf(PerceptronEntries); i = i + 1) begin
            sum = boundedPlus(sum, (history[i-1] ? extend(weight[i]) : extend(-weight[i]))); // Think about hardware this implies. - log (128) = 9 deep?
        end
        for (Integer i = 0; i < valueOf(PerceptronGHistEntries); i = i + 1) begin
            // TODO (RW): Should I be using a global bias?
            sum = boundedPlus(sum, ((gHist[i] == 1) ? extend(glob_weight[i]) : extend(-glob_weight[i])));
        end
        return sum;
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

                let index = getIndex(offsetPc(pc_reg, i));

                // In pred, most recent is correct
                PerceptronGHist globHist = global_history.history;                

                let sum = computePerceptronOutput(weights.sub(index), histories.sub(index), global_weights.sub(index), global_history);

                Bool taken = (sum >= 0);
                Bool forceTrain = (abs(sum) < fromInteger(trunc((1.93 * (fromInteger(valueOf(PerceptronEntries)))) + 14)));

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
                        index: index,
                        train: forceTrain
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
        let forceTrain = train.train;

        // update history if mispred
        // TODO (RW): Does this work for cases where two predictions have been made in the same cycle?
        // How to resolve - passing index? 
        // TODO (RW): Does this also cause issues where update is called much later? This would be harder to fix...
        if (mispred) begin
            PerceptronGHist newHist = truncate({pack(taken), train.gHist} >> 1);
            global_history.redirect(newHist);
        end 
    
        // Paper says threshold = 1.93 * branch history + 14. 
        // TODO (RW): Measure with and without?
        
        
        let local_hist = histories.sub(index);
        PerceptronWeights local_weights = weights.sub(index);
        PerceptronGWeights g_weights = global_weights.sub(index);
        
        // Train bias
        // TODO (RW): Should this be guarded behind the training threshold?
        local_weights[0] = boundedPlus(local_weights[0], ((taken) ? 1 : -1));
        // TODO (RW): Why isn't this updating (sits at 0) (check!)

        // Train local and global weights
        
        // Bool localCorrelationPos, globCorrelationPos;
        // Int#(8) localInc, globInc;
        if (mispred || forceTrain) begin
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
        end
        
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
    method Action flush if (!resetHist);
        // Local hist, weights, gweights handled by resetHist
        resetHist <= True;
        // GHist
        PerceptronGHist empty = 0;
        global_history.redirect(empty);
    endmethod

    // Not sure if this is some special method meaning `readable', or if it is just a normal read. Test with UTs!
    method flush_done = !resetHist._read;
endmodule

