import java.io.FileWriter;
import java.util.Scanner;

public class Nnet {
    double[][][] connection_weights;
    double[][] biases;
    int num_layers;
    int[] num_nodes;
    public Nnet(int[] numnodes){
        this.num_layers = numnodes.length;
        this.connection_weights = new double[this.num_layers-1][][];
        this.biases = new double[num_layers-1][];

        for(int layer=0; layer<num_layers-1; layer++){
            this.biases[layer] = new double[numnodes[layer+1]];
            for(int i=0; i<numnodes[layer+1]; i++){
                this.biases[layer][i] = 0.01*Math.random();
            }
            this.connection_weights[layer] = new double[numnodes[layer]][numnodes[layer+1]];
            for(int i=0; i<numnodes[layer]; i++){
                for(int j=0; j<numnodes[layer+1]; j++){
                    this.connection_weights[layer][i][j] = Math.random()-0.5;
                }
            }
        }
        this.num_nodes = numnodes;
    }

    public double[] compute_output_values(double[] input){
        double[][] values = new double[this.num_layers][];
        for(int i=0; i<this.num_layers; i++){
            values[i] = new double[this.num_nodes[i]];
        }
        values[0] = input;
        double sum;
        for(int layer=1; layer<this.num_layers; layer++){
            for(int i=0; i<this.num_nodes[layer]; i++){
                sum = 0;
                for(int j=0; j<this.num_nodes[layer-1]; j++){
                    sum += values[layer-1][j] * this.connection_weights[layer-1][j][i];
                }
                values[layer][i] = sigmoid(sum+biases[layer-1][i]);
            }
        }
        return values[num_layers-1];
    }

    public double[][] compute_all_values(double[] input){
        double[][] computed_values = new double[num_layers][];
        computed_values[0] = input;
        double sum;
        for(int layer=1; layer<this.num_layers; layer++){
            computed_values[layer] = new double[num_nodes[layer]];
            for(int i=0; i<this.num_nodes[layer]; i++){
                sum = 0;
                for(int j=0; j<this.num_nodes[layer-1]; j++){
                    sum += computed_values[layer-1][j] * this.connection_weights[layer-1][j][i];
                }
                computed_values[layer][i] = sigmoid(sum+biases[layer-1][i]);
            }
        }
        return computed_values;
    }

    /* raw_values[num_layers]
     * layer_errors[num_layers]
     * computed_values[num_layers]
     * biases[num_layers-1]
     * connection_weights[num_layers-1]
     */

    public void train(double[] input, double[] target_output, double learning_rate){
        double[][] computed_values = compute_all_values(input);

        // layer_errors[][] is dError/dRaw
        double[][] layer_errors = new double[this.num_layers][];
        layer_errors[this.num_layers-1] = new double[this.num_nodes[this.num_layers-1]];
        for(int i = 0; i<this.num_nodes[this.num_layers-1]; i++){
            layer_errors[this.num_layers-1][i] = (computed_values[this.num_layers-1][i] - target_output[i]) * sigmoid_derivative(computed_values[this.num_layers-1][i]);
        }
        //Backpropogation
        double error; // dError/dRaw
        for(int layer = this.num_layers-2; layer > 0; layer--){
            layer_errors[layer] = new double[this.num_nodes[layer]];
            //Loop through all nodes in layer
            for(int i = 0; i<this.num_nodes[layer]; i++){
                error = 0.0;
                //Loop through all nodes in layer+1
                for(int j = 0; j<this.num_nodes[layer+1]; j++){
                    error += layer_errors[layer+1][j] * connection_weights[layer][i][j];
                }
                layer_errors[layer][i] = error * sigmoid_derivative(computed_values[layer][i]);
            }
        }
        //Update weights and biases
        //Note that "error" = dError/dRaw
        for(int layer = this.num_layers-2; layer >= 0; layer--){
            for(int j = 0; j<this.num_nodes[layer+1]; j++){
                for(int i = 0; i<this.num_nodes[layer]; i++){
                    connection_weights[layer][i][j] -= layer_errors[layer+1][j] * computed_values[layer][i] * learning_rate;
                }
                biases[layer][j] -= layer_errors[layer+1][j] * learning_rate;
            }
        }
    }

    public void write_to_file(String filename){
        FileWriter w = new FileWriter(filename);
        w.write("Hello");
    }

    public static double sigmoid(double x){
        return 1/(1+Math.exp(-x));
    }

    // Input is a pre-sigmoid-activated value
    public static double sigmoid_derivative(double sigmoid_value){
        return sigmoid_value * (1-sigmoid_value);
    }

    public static double[] generate_random_array(int arr_length){
        double[] output = new double[arr_length];
        for(int i=0; i<arr_length; i++){
            output[i] = Math.random();
        }
        return output;
    }
}
