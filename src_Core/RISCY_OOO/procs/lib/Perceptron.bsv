import Types::*;
import ProcTypes::*;
import RegFile::*;
import Vector::*;
import BrPred::*;

export PerceptronTrainInfo;
export mkPerceptron;
export PerceptronEntries;
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

typedef SizeOf#(Addr) AddrWidth; // Numeric: Number of bits in an address.
typedef TExp#(AddrWidth) AddrRange; // Numeric: Number of addresses in the range.
typedef TDiv#(AddrRange, 4) PerceptronCount; // Numeric: Number of perceptrons - depends on hash function.
typedef TLog#(PerceptronCount) PerceptronsRegIndexWidth; // Numeric: Number of bits to be used for indexing the Regfile of perceptrons. TODO (RW): Increase by 1 to be big enough for weights!
typedef Bit#(PerceptronsRegIndexWidth) PerceptronsRegIndex; // Value: Bits used as the index for the Regfile.
 
typedef PerceptronsRegIndex PerceptronTrainInfo;

typedef Vector#(PerceptronEntries, Bool) PerceptronHistory;
typedef Vector#(TAdd#(PerceptronEntries, 1), Int#(8)) PerceptronWeights;

interface PerceptronHistorian; // Not stateful
    method PerceptronHistory update(PerceptronHistory hist, Bool taken);
    method Bool get(PerceptronHistory hist, Integer index); // TODO (RW): Instead of Integer, want PerceptronIndex. What will it do if you call with too big a value?
    method PerceptronHistory initHist();
endinterface

module mkPerceptronHistorianShiftReg(PerceptronHistorian);
    // TODO (RW): Could define another implementation which uses a head pointer and overwrites oldest value on update.

    method PerceptronHistory update(PerceptronHistory hist, Bool taken);
        // shift all history values down one, add new value at the top.
        for (Integer i = 1; i < valueOf(PerceptronEntries); i = i + 1) begin
            hist[i] = hist[i - 1];
        end
        hist[0] = taken;
        return hist; // Can't update history in place as it can't be a reg.
    endmethod

    method Bool get(PerceptronHistory hist, Integer index);
        return hist[index];
    endmethod

    method PerceptronHistory initHist;
        PerceptronHistory hist = replicate(False);
        return hist;
    endmethod
endmodule

(* synthesize *)
module mkPerceptron(DirPredictor#(PerceptronTrainInfo));
    PerceptronHistorian ph <- mkPerceptronHistorianShiftReg;
    RegFile#(PerceptronsRegIndex, PerceptronHistory) histories <- mkRegFileWCF(0,fromInteger(valueOf(PerceptronCount)-1));
    Reg#(PerceptronHistory) global_history <- mkRegU;
    RegFile#(PerceptronsRegIndex, PerceptronWeights) weights <- mkRegFileWCF(0,fromInteger(valueOf(PerceptronCount)-1)); 
    RegFile#(PerceptronsRegIndex, PerceptronWeights) global_weights <- mkRegFileWCF(0,fromInteger(valueOf(PerceptronCount)-1)); 
    
    Reg#(Addr) pc_reg <- mkRegU;
    // TODO (RW): Decide max weight size and prevent overflow. 8 suggested in paper.
    // TODO (RW): Use some additional local weights for global history? Could be second reg file, or could double size of weights reg file.
    // TODO (RW): Allow size of global history to be different to that of each local history
    

    Reg#(PerceptronsRegIndex) i <- mkReg(0);
    Reg#(Bool) resetHist <- mkReg(True);
    
    rule initHistory(resetHist);
        if (i == 0) begin
            histories.upd(i, ph.initHist());
            weights.upd(i, replicate(0));
            global_history <= ph.initHist();
            global_weights.upd(i, replicate(0));
            i <= i + 1;
        end
        else if (i < fromInteger(valueOf(PerceptronIndexWidth))) begin // Should be PerceptronCount, not PerceptronIndexWidth. Should check if ON THE LAST CASE, not past it.
            histories.upd(i, ph.initHist());
            weights.upd(i, replicate(0));
            global_weights.upd(i, replicate(0));
            i <= i + 1;
        end
        else begin
            i <= 0;
            resetHist <= False;
        end

        // TODO (RW): Should this be done in a separate rule?
        // TODO (RW): Make i be state, and have rule just reset histories[i]. Need to clear resetHist afterwards.
        
        // May need to guard things on not resetHist -> method stuff on history can only be done if not resetHist.
    endrule

    function PerceptronsRegIndex getIndex(Addr pc);
        return truncate(pc >> 2);
    endfunction

    // Function to compute the perceptron output
    function Bool computePerceptronOutput(PerceptronWeights weight, PerceptronHistory history, PerceptronWeights glob_weight, PerceptronHistory global_hist);
        Int#(16) sum = extend(weight[0]); // Bias weight - TODO (RW): check this can't overflow.
        for (Integer i = 1; i < valueOf(PerceptronEntries); i = i + 1) begin // TODO (RW): check loop boundary
            sum = sum + (history[i] ? extend(weight[i]) : extend(-weight[i])); // Think about hardware this implies. - log (128) = 9 deep?
            // TODO (RW): Add parameter to choose how much to use global history (multiplier)
            sum = sum + (global_hist[i] ? extend(glob_weight[i]) : extend(-glob_weight[i]));
        end
        return sum >= 0;
    endfunction

    // Interface for each perceptron in the table
    Vector#(SupSize, DirPred#(PerceptronTrainInfo)) predIfc;
    for(Integer i = 0; i < valueOf(SupSize); i = i+1) begin
        predIfc[i] = (interface DirPred;
            method ActionValue#(DirPredResult#(PerceptronTrainInfo)) pred;
                let index = getIndex(offsetPc(pc_reg, i));
                Bool taken = computePerceptronOutput(weights.sub(index), histories.sub(index), global_weights.sub(index), global_history);
                return DirPredResult {
                    taken: taken,
                    train: index
                };
            endmethod
        endinterface);
    end

    
    method nextPc = pc_reg._write;

    interface pred = predIfc;

    
    method Action update(Bool taken, PerceptronTrainInfo train, Bool mispred); 
        // TODO (RW): Only train if below training threshold. Paper says threshold = 1.93 * branch history + 14.
        
        let index = train; // already hashed
        let local_hist = histories.sub(index);
        PerceptronWeights local_weights = weights.sub(index);
        
        // Increment bias if taken, else decrement
        local_weights[0] = (taken) ? local_weights[0] + 1 : local_weights[0] - 1;

        for (Integer i = 1; i < valueOf(PerceptronEntries); i = i + 1) begin
            local_weights[i] = local_weights[i] + (taken == local_hist[i] ? 1 : -1);
        end

        // Update history
        local_hist = ph.update(local_hist, taken);
        global_history <= ph.update(global_history, taken);
    endmethod


    // Perceptron predictor also doesn't need to be flushed
    method flush = noAction;
    method flush_done = True;
endmodule

