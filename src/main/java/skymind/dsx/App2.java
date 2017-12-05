package skymind.dsx;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * Hello world!
 *
 */
public class App2
{
    private static Logger log = LoggerFactory.getLogger(App2.class);

    public static void main( String[] args ) throws Exception {
        BasicConfigurator.configure();
        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 0;
        char delimiter = ',';
        int labelIndex = 4;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
        int numClasses = 3;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
        int batchSize = 150;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)


        RecordReader trainrecordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        trainrecordReader.initialize(new FileSplit(new File("iris_train.txt")));

        RecordReader testrecordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        testrecordReader.initialize(new FileSplit(new File("iris_test.txt")));

        // Load all the train to calculate normalizer
        DataSetIterator fulliterator = new RecordReaderDataSetIterator(trainrecordReader, 98, labelIndex, numClasses);

        DataSetIterator iterTrain = new RecordReaderDataSetIterator(trainrecordReader, batchSize, labelIndex, numClasses);
        DataSetIterator iterTest = new RecordReaderDataSetIterator(testrecordReader, batchSize, labelIndex, numClasses);

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network

        NormalizerStandardize preProcessor = new NormalizerStandardize();
        preProcessor.fit(fulliterator);
        iterTrain.setPreProcessor(preProcessor);
        iterTest.setPreProcessor(preProcessor);

        DataSet trainingData = iterTrain.next();
        DataSet testData = iterTest.next();

        final int numInputs = 4;
        int outputNum = 3;
        int iterations = 1000;
        long seed = 6;


        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.1)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(3)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(outputNum).build())
                .backprop(true).pretrain(false)
                .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        model.fit(trainingData);

        //evaluate the model on the test set
        Evaluation eval = new Evaluation(3);
        INDArray output = model.output(testData.getFeatureMatrix());
        eval.eval(testData.getLabels(), output);
        log.info(eval.stats());
        log.info("*******************************th*******************");
    }
}
