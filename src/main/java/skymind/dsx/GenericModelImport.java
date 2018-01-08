package skymind.dsx;

/**
 * Created by tomhanlon on 12/29/17.
 */
import org.apache.log4j.BasicConfigurator;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


public class GenericModelImport {
    public static void main(String[] args) throws Exception{
        BasicConfigurator.configure();



        String kerasModelfromKerasExport = "MyModel.h5";


        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(kerasModelfromKerasExport);


        System.out.println("Model Configuration");
        String modelconfig = model.conf().toJson();
        System.out.println(modelconfig);



    }
}
