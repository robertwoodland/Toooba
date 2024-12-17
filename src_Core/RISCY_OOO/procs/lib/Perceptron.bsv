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
typedef 128 PerceptronEntries; // Entries per perceptron
typedef Bit#(TLog#(PerceptronEntries)) PerceptronIndex; // Index for perceptron

typedef PerceptronIndex PerceptronTrainInfo;


(* synthesize *)
module mkPerceptron(DirPredictor#(PerceptronTrainInfo));
    RegFile#(PerceptronIndex, Bit#(2)) history <- mkRegFileWCF(0,fromInteger(valueOf(PerceptronEntries)-1));
    RegFile#(PerceptronIndex, Bit#(2)) weights <- mkRegFileWCF(0,fromInteger(valueOf(PerceptronEntries)-1));

    function PerceptronIndex getIndex(Addr pc);
        return truncate(pc >> 2);
    endfunction

    // Function to compute the perceptron output
    function Bool computePerceptronOutput(Vector#(PerceptronEntries, Int#(8)) weight, Vector#(PerceptronEntries, Bool) history);
        Int#(16) sum = weight[0]; // Bias weight
        for (Integer i = 1; i < valueof(PerceptronEntries); i = i + 1) begin
            sum = sum + (history[i] ? weight[i] : -weight[i]);
        end
        return sum >= 0;
    endfunction

    // Interface for each perceptron in the table
    Vector#(SupSize, DirPred#(PerceptronTrainInfo)) predIfc;
    for(Integer i = 0; i < valueof(SupSize); i = i+1) begin
        predIfc[i] = (interface DirPred;
            method ActionValue#(DirPredResult#(PerceptronTrainInfo)) pred;
                let index = getIndex(offsetPc(pc_reg, i));
                Bool taken = computePerceptronOutput(weights[i], history[i]);
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
    
        let index = train;
        // Increment bias if taken, else decrement
        let bias = history.sub(index)[0];
        history.sub(index)[0] = (taken) ? bias + 1 : bias - 1;

        // TODO (RW): Redefine for perceptron
        let current_hist = history.sub(index);
        Bit#(2) next_hist;
        if(taken) begin
            next_hist = (current_hist == 2'b11) ? 2'b11 : current_hist + 1;
        end else begin
            next_hist = (current_hist == 2'b00) ? 2'b00 : current_hist - 1;
        end
        history.upd(index, next_hist); 
    endmethod


    // Perceptron predictor also doesn't need to be flushed
    method flush = noAction;
    method flush_done = True;
endmodule

