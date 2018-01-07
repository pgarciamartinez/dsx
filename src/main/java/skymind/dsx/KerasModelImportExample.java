package skymind.dsx;

/**
 * Created by tomhanlon on 12/29/17.
 */
import org.apache.log4j.BasicConfigurator;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;




public class KerasModelImportExample {
    public static void main(String[] args) throws Exception{
        BasicConfigurator.configure();


        //Path to Saved Model and weights


        String kerasModelfromKerasExport = "Keras_export_full_iris_model";
        /*
        Create a MultiLayerNetwork from the saved model
         */


        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(kerasModelfromKerasExport);
        /*
        The Model trained on Iris data, 4 fields
        Sepal Length, Sepal Width, Petal Length, Petal Width
        When asked to predict the class for the following input

        prediction = model.predict(numpy.array([[4.6,3.6,1.0,0.2]]));

        Output was...
        [[ 0.92084521  0.13397516  0.03294737]]

        To verify the output is proper for the loaded model test with the same data
        Input [4.60, 3.60, 1.00, 0.20]
        Output[0.92, 0.13, 0.03]
         */

        INDArray myArray = Nd4j.zeros(1, 4); // one row 4 column array
        myArray.putScalar(0,0, 4.6);
        myArray.putScalar(0,1, 3.6);
        myArray.putScalar(0,2, 1.0);
        myArray.putScalar(0,3, 0.2);

        INDArray output = model.output(myArray);
        System.out.println("First Model Output");
        System.out.println(myArray);
        System.out.println(output);

    }
}
