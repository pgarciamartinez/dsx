package skymind.dsx;

import org.apache.log4j.BasicConfigurator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * Created by tomhanlon on 12/4/17.
 */
public class UCISequenceClassificationSpark {

    private static final Logger log = LoggerFactory.getLogger(UCISequenceClassificationSpark.class);

    public static void main(String[] args) throws Exception {
        BasicConfigurator.configure();

        // Number of examples per Spark worker
        int batchSizePerWorker = 20;

        // Total Number of passes through the data
        int numEpochs = 20;

        // Set up Spark Configuration
        SparkConf sparkConf = new SparkConf();
        sparkConf.setAppName("DL4J Spark RNN Example");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);




        // Unzip the data that has been shipped to Spark Context
        ZipFile zipFile = new ZipFile("uci.zip");
        Enumeration<?> enu = zipFile.entries();
        while (enu.hasMoreElements()) {
            ZipEntry zipEntry = (ZipEntry) enu.nextElement();

            String name = zipEntry.getName();
            long size = zipEntry.getSize();
            long compressedSize = zipEntry.getCompressedSize();
            System.out.printf("name: %-20s | size: %6d | compressed size: %6d\n",
                    name, size, compressedSize);

            // Do we need to create a directory ?
            File file = new File(name);
            if (name.endsWith("/")) {
                file.mkdirs();
                continue;
            }

            File parent = file.getParentFile();
            if (parent != null) {
                parent.mkdirs();
            }

            // Extract the file
            InputStream is = zipFile.getInputStream(zipEntry);
            FileOutputStream fos = new FileOutputStream(file);
            byte[] bytes = new byte[1024];
            int length;
            while ((length = is.read(bytes)) >= 0) {
                fos.write(bytes, 0, length);
            }
            is.close();
            fos.close();

        }
        zipFile.close();
        // End of Code Specific to Unziping Data Archive


        // Create Sequence Readers to Read Data
        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();

        // initialize Record Reader with path to Data
        trainFeatures.initialize(new NumberedFileInputSplit(new File("uci/train/features").getAbsolutePath().toString() + "/%d.csv", 0, 449));
        //trainFeatures.initialize(new NumberedFileInputSplit(new F, 0, 449));
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit(new File("uci/train/labels").getAbsolutePath().toString() + "/%d.csv", 0, 449));

        int miniBatchSize = 20;
        int numLabelClasses = 6;
        DataSetIterator trainDataiter = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numLabelClasses,
                false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        // Normalize the data
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainDataiter);
        trainDataiter.setPreProcessor(normalizer);


        // Same process for Test Data as Train Data
        SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
        testFeatures.initialize(new NumberedFileInputSplit(new File("uci/test/features").getAbsolutePath().toString() + "/%d.csv", 0, 149));
        SequenceRecordReader testLabels = new CSVSequenceRecordReader();
        testLabels.initialize(new NumberedFileInputSplit(new File("uci/test/labels").getAbsolutePath().toString() + "/%d.csv", 0, 149));
        DataSetIterator testDataiter = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, numLabelClasses,
                false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
        testDataiter.setPreProcessor(normalizer);

        // Get that Data into a Spark RDD
        // First as List
        List<DataSet> trainDataList = new ArrayList<>();
        List<DataSet> testDataList = new ArrayList<>();
        while (trainDataiter.hasNext()) {
            trainDataList.add(trainDataiter.next());
        }
        while (testDataiter.hasNext()) {
            testDataList.add(testDataiter.next());
        }

        // List to RDD
        JavaRDD<DataSet> trainData = sc.parallelize(trainDataList);
        JavaRDD<DataSet> testData = sc.parallelize(testDataList);


        // Build a Neural Net Configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.9))
                .learningRate(0.005)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(0.5)
                .list()
                .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(10).nOut(numLabelClasses).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();


        // Configure a Training Master for Spark
        // Training Master coordinates data sharing among workers
        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)    //Each DataSet object: contains (by default) 32 examples
                .averagingFrequency(5)
                .workerPrefetchNumBatches(0)
                .rddTrainingApproach(RDDTrainingApproach.Direct)//Async prefetching: 2 examples per worker
                .batchSizePerWorker(batchSizePerWorker)
                .build();

        // Wrap the Neural Net Configuration for Spark Distributed Execution
        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf, tm);

        // Configure a string for printing Accuracy and F1 score
        String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";


        // Train the Neural Network
        for (int i = 0; i < numEpochs; i++) {
            sparkNet.fit(trainData);

            //Evaluate on the test set:

            //Evaluation evaluation = sparkNet.evaluate(testData);
            Evaluation evaluation = sparkNet.doEvaluation(testData, 10, new Evaluation(6))[0];

            log.info(String.format(str, i, evaluation.accuracy(), evaluation.f1()));

            testDataiter.reset();
            trainDataiter.reset();

            log.info("Completed Epoch {}", i);
            log.info("############ STATS ##########");
            log.info(evaluation.stats());
        }



        System.out.println("##############DONE############");
        log.info("----- Example Complete -----");

    }

}