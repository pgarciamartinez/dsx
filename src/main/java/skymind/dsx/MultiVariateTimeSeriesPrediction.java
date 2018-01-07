package skymind.dsx;

/**
 * Created by tomhanlon on 12/29/17.
 */
import org.apache.log4j.BasicConfigurator;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;


public class MultiVariateTimeSeriesPrediction {
    public static void main(String[] args) throws Exception{
        BasicConfigurator.configure();


        //Path to Saved Model and weights


        String kerasModelfromKerasExport = "pollution.h5";
        /*
        Create a MultiLayerNetwork from the saved model
         */
        boolean enforceTrainingConfig = false; // Model can not trained further

        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(kerasModelfromKerasExport,enforceTrainingConfig);


        INDArray input = Nd4j.create(new double[]{0.12977867,0.35294122,0.24590163,0.52727318,0.66666669,0.00229001,0.,0.,0.14889336,0.36764708,0.24590163,0.52727318,0.66666669,0.00381099,0.,0.,0.15995975,0.42647061,0.22950819,0.54545403,0.66666669,0.00533197,0.,0.},new int[]{3,8});
       
        INDArray output = model.output(input);
        System.out.println("######## OUTPUT #########");
        System.out.println(output);


    }
}
