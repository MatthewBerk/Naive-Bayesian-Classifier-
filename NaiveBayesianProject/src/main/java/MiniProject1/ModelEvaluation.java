/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MiniProject1;

import weka.core.Instances;
import weka.core.Instance;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Matthew Berkman
 */
public class ModelEvaluation {

    private String CSVFilePath;
    private Instances emptyInstance;

    private int truePositives = 0;
    private int falsePositives = 0;
    private int falseNegatives = 0;
    private int trueNegatives = 0;

    double modelsAccuracy = 0;
    double positiveLabelPrecision;
    double positiveLabelRecall;
    double positiveLabelF1;
    double negativeLabelPrecision;
    double negativeLabelRecall;
    double negativeLabelF1;
    
    String positiveClassLabel;
    String negativeClassLabel;
    
    String[] missingValueStrategies;
    String[] continuousAttributeStrategies;
    

    public ModelEvaluation(String filePath) {
        CSVFilePath = filePath;
        emptyInstance = loadDataSet();
        emptyInstance.delete();
    }
    
    
    public void setMissingValueStrategies(String[] strategyForEachAttrbuteMissingValue){
        missingValueStrategies = strategyForEachAttrbuteMissingValue;
        
    }
    
    public void setContinuousAttributeStrategies(String[] strategies){
        continuousAttributeStrategies = strategies;
    }

  
    public void performKFoldValidation(int k) {

        Instances entireDataSet = loadDataSet();
   
        for (int iterations = 0; iterations < k; iterations++) {//iterations represents which bin will be test data.
            Instances dataClone = new Instances(entireDataSet);//Just to make sure original data object doesn't get altered so don't need to re-read file
            Instances[] dataSeperated = createDataBins(dataClone, k);//splits datatset into k bins with equal values excluding last one which holds whatever was left.
           
            Instances trainingDataSet = new Instances(emptyInstance);
            for (int index = 0; index < k; index++) {
                if (index == iterations) {
                    continue;
                }
                Enumeration entries = dataSeperated[index].enumerateInstances();
                //Need to recombine all the instances that will be used for training since did not design my model
                // to take in different training sets one at a time. Also that would effect calculation for misisng values such as mean.
                while (entries.hasMoreElements()) {
                    trainingDataSet.add((Instance) entries.nextElement());
                }
            }

            NaiveBayes trainedModel = trainNaiveBayesModel(trainingDataSet);

            //now to run test since model is built.
            List<String> modelsPredictions = trainedModel.testModel(dataSeperated[iterations],missingValueStrategies);

            int[][] confusionMatrix = constructConfusionMatrix(dataSeperated[iterations], modelsPredictions);

            //Keeps track of the performance of each k fold since need to measure performance after all k-folds are done.
            for (int columnIndex = 0; columnIndex < confusionMatrix.length; columnIndex++) {
                for (int rowIndex = 0; rowIndex < confusionMatrix[columnIndex].length; rowIndex++) {
                    //Rough version of code to handle any size confusion matrix. Focused on just 2x2 though.
                    if (columnIndex <= (confusionMatrix.length / 2.0) - 1 && columnIndex == rowIndex) {
                        truePositives += confusionMatrix[columnIndex][rowIndex];
                    } else if (columnIndex == rowIndex) {
                        trueNegatives += confusionMatrix[columnIndex][rowIndex];
                    } else if (columnIndex < rowIndex) {
                        falsePositives += confusionMatrix[columnIndex][rowIndex];
                    } else {
                        falseNegatives += confusionMatrix[columnIndex][rowIndex];
                    }
                }
            }
        }

        //now to calculate performance for both class labels
        modelsAccuracy = calculateAccuracy();
        positiveLabelPrecision = calculatePrecision(truePositives,falsePositives);
        positiveLabelRecall = calculateRecall(truePositives, falseNegatives);
        positiveLabelF1 = calculateF1Measure(positiveLabelPrecision, positiveLabelRecall);
        
        negativeLabelPrecision = calculatePrecision(trueNegatives,falseNegatives);
        negativeLabelRecall = calculateRecall(trueNegatives,falsePositives);
        negativeLabelF1 = calculateF1Measure(negativeLabelPrecision, negativeLabelRecall);
          
    }

    //Loads data from file specified by CSVFilePath and puts it in a Weka Instances
    public Instances loadDataSet() {
        CSVLoader loader = new CSVLoader();
        try {
            loader.setSource(new File(CSVFilePath));
            return loader.getDataSet();
        } catch (IOException ex) {
            Logger.getLogger(ModelEvaluation.class.getName()).log(Level.SEVERE, null, ex);
        }
        return null;
    }

    private NaiveBayes trainNaiveBayesModel(Instances trainingData) {
        NaiveBayes aModel = new NaiveBayes(trainingData, " ?");

        aModel.handleMissingValues(missingValueStrategies, aModel.theData);

        aModel.constructClassValuesCounter();
        aModel.constructValuesCounter();

        //Setups up either binning or gausian for the numerical attributes.
        aModel.handleContinuousAttributes(continuousAttributeStrategies, aModel.theData);   

        aModel.constructModel();

        return aModel;
    }

    //splits theData "equally" among numBin, though last bin likely to have more values due to leftovers
    //THIS METHOD IS NOT binning method for continuous attributes, it is binning method for k-fold cross validation
    private Instances[] createDataBins(Instances theData, int numBin) {
        Instances[] holder = new Instances[numBin];
        int dataSize = theData.size();
        int binSize = dataSize / numBin;//last bin will have more data then rest due to leftovers.

        Enumeration entries = theData.enumerateInstances();//Now enumerating through all entries in theData
        for (int binNumber = 0; binNumber < numBin; binNumber++) {
            Instances tempInstance = new Instances(emptyInstance);
            if (binNumber + 1 == numBin) {//at last bin so rest of data goes in it
                while (entries.hasMoreElements()) {
                    tempInstance.add((Instance) entries.nextElement());
                }
            } else {  
                for (int atEntry = binNumber * binSize; atEntry < (binNumber + 1) * binSize; atEntry++) {
                    tempInstance.add((Instance) entries.nextElement());
                }
            }
            holder[binNumber] = tempInstance;

        }
        return holder;
    }

    //Creates the confusion matrix to help determine True positives, Fp, FN, TN so can calculate measures such as preicsion.
    public int[][] constructConfusionMatrix(Instances testData, List<String> predictions) {
        Enumeration classValues = testData.attribute(testData.numAttributes() - 1).enumerateValues();
        List<String> labelValues = new ArrayList<>();

        while (classValues.hasMoreElements()) {
            labelValues.add(classValues.nextElement().toString());
        }
        positiveClassLabel = labelValues.get(0);//originally set out to design ModelEvaluation to handle different kinds of datasets, where
        // class labels is more than 2, but realized it would get very complex and wouldn't have an efficient way of testing it, so now assuming 
        // just have two class labels.
        negativeClassLabel = labelValues.get(1);

        int[][] confusionMatrx = new int[labelValues.size()][labelValues.size()];

        for (int index = 0; index < predictions.size(); index++) {
            Instance anEntry = testData.get(index);
            String actualValue = anEntry.stringValue(testData.numAttributes() - 1);

            int guessLabelIndex = labelValues.indexOf(predictions.get(index));
            int actualValueIndex = labelValues.indexOf(actualValue);

            confusionMatrx[actualValueIndex][guessLabelIndex] += 1;
        }

        return confusionMatrx;

    }

    public double calculateAccuracy() {
        return ((truePositives + trueNegatives) * 1.0) / (truePositives + trueNegatives + falsePositives + falseNegatives);
    }

    //tp would be number correct for a certain class label, fp would be incorrect guessed for a class label
    public double calculatePrecision(int tp, int fp) {
        return (tp * 1.0) / (tp + fp);
    }

    //tp would be number correct for a certain class label, fn would be incorrect guessed for the other class label
    public double calculateRecall(int tp, int fn) {
        return (tp * 1.0) / (tp + fn);
    }

    public double calculateF1Measure(double precision, double recall) {
        return 2*((precision*recall)/(precision+recall));
        
    }

    public String getPositiveClassLabel(){
        return positiveClassLabel;
    }
    
    public String getNegativeClassLabel(){
        return negativeClassLabel;
    }
    
    public double getModelsAccuracy(){
        return modelsAccuracy;
    }
    
    public double getPositiveLabelPrecision(){
        return positiveLabelPrecision;
    }
    
    public double getPositiveLabelRecall(){
        return positiveLabelRecall;
    }
    
    public double getPositiveLabelF1(){
        return positiveLabelF1;
    }
    
     public double getNegativeLabelPrecision(){
        return negativeLabelPrecision;
    }
    
    public double getNegativeLabelRecall(){
        return negativeLabelRecall;
    }
    
    public double getNegativeLabelF1(){
        return negativeLabelF1;
    }
    
 
}
