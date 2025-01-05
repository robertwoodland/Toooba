import Types::*;
import ProcTypes::*;
import RegFile::*;
import Vector::*;
import BrPred::*;

export PerceptronTrainInfo;
export mkPerceptron;
export PerceptronEntries;
export PerceptronIndex;

// Local Perceptron Typedefs
typedef 128 PerceptronEntries; // Size of perceptron (length of history) - typically 4 to 66 depending on hardware budget.
typedef Bit#(PerceptronEntries / 4) PerceptronIndex; // Number of perceptrons - depends on hash function. This should be a log?! Can't just divide PerceptronEntries as it's a type. Valueof(). Or tdiv / tadd / tsub.

typedef PerceptronIndex PerceptronTrainInfo;

interface PerceptronHistory#(numeric type PerceptronEntries); // TODO (RW): Don't need to parameterise interface based on size.
    method Action update(Bool taken);
    method Bool get(Int index); // TODO (RW): Instead of int, want something valueof(perceptronindex). What will it do if you call with too big a value?
endinterface

module mkPerceptronHistory(PerceptronHistory#(PerceptronEntries)); // TODO (RW): Rename this to mkPerceptronHistoryShiftReg.
    Vector#(PerceptronEntries, Bool) history;

    // TODO (RW): Could define another implementation which uses a head pointer and overwrites oldest value on update.

    method Action update(Bool taken);
        // shift all history values down one, add new value at the top.
        for (Integer i = 1; i < valueof(PerceptronEntries); i = i + 1) begin
            history[i] = history[i - 1];
        end
        history[0] = taken;
    endmethod

    method Bool get(Int index);
        return history[index];
    endmethod
endmodule

(* synthesize *)
module mkPerceptron(DirPredictor#(PerceptronTrainInfo));
    RegFile#(PerceptronIndex, PerceptronHistory) histories <- mkRegFileWCF(0,fromInteger(valueOf(PerceptronIndex)-1)); // TODO (RW): Instantiate with mkPerceptronHistory somehow
    PerceptronHistory global_history <- mkPerceptronHistory(PerceptronEntries);
    RegFile#(PerceptronIndex, Vector#(PerceptronEntries, Int#(8))) weights <- mkRegFileWCF(0,fromInteger(valueOf(PerceptronIndex)-1)); 
    RegFile#(PerceptronIndex, Vector#(PerceptronEntries, Int#(8))) global_weights <- mkRegFileWCF(0,fromInteger(valueOf(PerceptronIndex)-1)); 
    // TODO (RW): Decide max weight size and prevent overflow. 8 suggested in paper.
    // TODO (RW): Use some additional local weights for global history? Could be second reg file, or could double size of weights reg file.
    // TODO (RW): Allow size of global history to be different to that of each local history


    function PerceptronIndex getIndex(Addr pc);
        return truncate(pc >> 2);
    endfunction

    // Function to compute the perceptron output
    function Bool computePerceptronOutput(Vector#(PerceptronEntries, Int#(8)) weight, Vector#(PerceptronEntries, Bool) history, Vector#(PerceptronEntries, Int#(8)) glob_weight, Vector#(PerceptronEntries, Bool) global_history);
        Int#(16) sum = weight[0]; // Bias weight - TODO (RW): check this can't overflow.
        for (Integer i = 1; i < valueof(PerceptronEntries); i = i + 1) begin // TODO (RW): check loop boundary
            sum = sum + (history.get(i) ? weight[i] : -weight[i]); // Think about hardware this implies. - log (128) = 9 deep?
            // TODO (RW): Use global history too 
            sum = sum + (global_history.get(i) ? glob_weight[i] : -glob_weight[i]);
        end
        return sum >= 0;
    endfunction

    // Interface for each perceptron in the table
    Vector#(SupSize, DirPred#(PerceptronTrainInfo)) predIfc;
    for(Integer i = 0; i < valueof(SupSize); i = i+1) begin
        predIfc[i] = (interface DirPred;
            method ActionValue#(DirPredResult#(PerceptronTrainInfo)) pred;
                // TODO (RW): Pass global weights through
                let index = getIndex(offsetPc(pc_reg, i));
                Bool taken = computePerceptronOutput(weights[i], histories.sub(i), global_weights[i], global_history);
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
        let local_weights = weights.sub(index);
        
        // Increment bias if taken, else decrement
        local_weights[0] = (taken) ? local_weights[0] + 1 : local_weights[0] - 1;

        for (Integer i = 1; i < valueof(PerceptronEntries); i = i + 1) begin
            local_weights[i] = local_weights[i] + (taken == local_hist[i] ? 1 : -1);
        end

        // Update history
        local_hist.update(taken);
        global_history.update(taken);
    endmethod


    // Perceptron predictor also doesn't need to be flushed
    method flush = noAction;
    method flush_done = True;
endmodule

