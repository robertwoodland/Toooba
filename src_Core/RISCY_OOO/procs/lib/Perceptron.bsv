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
typedef Bit#(PerceptronEntries / 4) PerceptronIndex; // Number of perceptrons - depends on hash function

typedef PerceptronIndex PerceptronTrainInfo;


(* synthesize *)
module mkPerceptron(DirPredictor#(PerceptronTrainInfo));
    // history[0] is the global history (currently).
    RegFile#(PerceptronIndex, Vector#(PerceptronEntries, Bit#(1))) histories <- mkRegFileWCF(0,fromInteger(valueOf(PerceptronIndex)-1));
    RegFile#(PerceptronIndex, Vector#(PerceptronEntries, Int#(8))) weights <- mkRegFileWCF(0,fromInteger(valueOf(PerceptronIndex)-1)); 
    // TODO (RW): Decide max weight size and prevent overflow. 8 suggested in paper.
    // TODO (RW): Change type of history to be FIFO.
    // TODO (RW): Use some additional local weights for global history? Could be second reg file, or could double size of weights reg file.

    function PerceptronIndex getIndex(Addr pc);
        return truncate(pc >> 2) + 1; // Add 1 because 0 is global history
    endfunction

    // Function to compute the perceptron output
    function Bool computePerceptronOutput(Vector#(PerceptronEntries, Int#(8)) weight, Vector#(PerceptronEntries, Bool) history);
        // y : output
        // w = weights[index] = weight
        // x = histories[index] = history
        // y = w[0] + sum(x[i] * w[i]) for i = 1 to n
        Int#(16) sum = weight[0]; // Bias weight - TODO (RW): check this can't overflow.
        for (Integer i = 1; i < valueof(PerceptronEntries); i = i + 1) begin // TODO (RW): check loop boundary
            sum = sum + (history[i] ? weight[i] : -weight[i]); // Think about hardware this implies. - log (128) = 9 deep?
        end
        return sum >= 0;
    endfunction

    // Interface for each perceptron in the table
    Vector#(SupSize, DirPred#(PerceptronTrainInfo)) predIfc;
    for(Integer i = 0; i < valueof(SupSize); i = i+1) begin
        predIfc[i] = (interface DirPred;
            method ActionValue#(DirPredResult#(PerceptronTrainInfo)) pred;
                let index = getIndex(offsetPc(pc_reg, i));
                Bool taken = computePerceptronOutput(weights[i], histories[i]);
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
        
        // t = taken
        // w_i = local_weights[i]
        // x_i = local_hist[i]
        // w_i = w_i + t * x_i
        // w_i = t ? (w_i + x_i) : (w_i - x_i)
        
        // Increment bias if taken, else decrement
        local_weights[0] = (taken) ? local_weights[0] + 1 : local_weights[0] - 1;

        for (Integer i = 1; i < valueof(PerceptronEntries); i = i + 1) begin
            local_weights[i] = local_weights[i] + (taken == local_hist[i] ? 1 : -1);
        end

        // TODO(RW): update histories (local + global) with new taken value (once structure decided).
    endmethod


    // Perceptron predictor also doesn't need to be flushed
    method flush = noAction;
    method flush_done = True;
endmodule

