/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MiniProject1;

import weka.core.converters.CSVLoader;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Attribute;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

import java.util.Enumeration;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.Map;
import java.util.HashMap;
import java.util.List;

/**
 *
 * @author Matthew Berkman
 */
public class NaiveBayes {

    public Instances theData;

    private Map<String, Integer> classValuesCounter;
    private Map<String, Map<String, Map<String, Integer>>> dataValuesCounter;//Will hold ocurrences of every attribute value when they are paired with a specific class label.
    private int classLabelIndex;//Index/column #  the class label is at.
    private String missingValueSymbol = "?";//What value (if any) was put in categorical attribtue slot when have missing value. Default is ?.

    private double[] attributeMeans;//categorical attributes don't get means so their value will remain 0.
    private String[] attributeMode;//numerical attributes don't get mode so their value will remain ""
    private double[] attributeMedian;

    private String[] contAtrStrategy;//The strategies want to use for continuous attributes, either number for width of bins, gaussian, or if categorical would put Category

    private double[] attributeVariances;//The variances of all numerical attributes that would be used by Gausian

    public NaiveBayes(String filePath) {
        CSVLoader loader = new CSVLoader();
        try {
            loader.setSource(new File(filePath));
            theData = loader.getDataSet();
            classLabelIndex = theData.numAttributes() - 1;//assuming last column is class label

            //these need to be done before we proceed to handle missing values since we use these calculations to assign value to missing value slot
            //I due not recalculate these after every missing value slot filled.
            attributeMeans = findAllAttributesMeans();
            attributeMode = findAllAttributesModes();
            attributeMedian = findAllAttributesMedians();

        } catch (IOException ex) {
            Logger.getLogger(NaiveBayes.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public NaiveBayes(String filePath, String missingValueIndicator) {
        this(filePath);
        missingValueSymbol = missingValueIndicator;
    }

    //Primarily used by ModelEvaluation
    public NaiveBayes(Instances givenData) {
        theData = givenData;
        classLabelIndex = theData.numAttributes() - 1;//assuming last column is class label

        //these need to be done before we proceed to handle missing values since we use these calculations to assign value to missing value slot
        //I due not recalculate these after every missing value slot filled.
        attributeMeans = findAllAttributesMeans();
        attributeMode = findAllAttributesModes();
        attributeMedian = findAllAttributesMedians();

    }

    //Primarily used by ModelEvaluation
    public NaiveBayes(Instances givenData, String missingValueIndicator) {
        this(givenData);
        missingValueSymbol = missingValueIndicator;
    }

    //Calculates the mean of every Numerical attribute with the current values present.
    private double[] findAllAttributesMeans() {
        double[] holder = new double[theData.numAttributes()];
        for (int index = 0; index < theData.numAttributes(); index++) {
            if (index == classLabelIndex) {
                continue;
            }
            holder[index] = calculateMean(index);
        }
        return holder;
    }

    //Calculates the mode of every Categorical attribute with the current values present.
    private String[] findAllAttributesModes() {
        String[] holder = new String[theData.numAttributes()];
        for (int index = 0; index < theData.numAttributes(); index++) {
            if (index == classLabelIndex) {//no reason to get mode of class label.
                continue;
            }
            holder[index] = calculateStringMode(index);
        }
        return holder;
    }

    //Calculates the median of every Numerical attribute with the current values present.
    private double[] findAllAttributesMedians() {
        double[] holder = new double[theData.numAttributes()];
        for (int index = 0; index < theData.numAttributes(); index++) {
            if (index == classLabelIndex) {//no reason to get median of class label
                continue;
            }
            holder[index] = findMedian(index);
        }
        return holder;
    }

    //Handles misisng values in all attributes whether no value present (NaN) or has missingValueSymbol (for categorical)
    public void handleMissingValues(String[] attributeStrategy, Instances temptData) {
        Enumeration entries = temptData.enumerateInstances();//enumerating through all values in temptData
        while (entries.hasMoreElements()) {
            Instance entry = (Instance) entries.nextElement();
            for (int index = 0; index < entry.numAttributes(); index++) {
                if (index == classLabelIndex) {
                    continue;//assuming we have no missing values for class label of training data
                }
                Double temp = entry.value(index);//categorical attributes would have some random number for temp which is why have temp2
                String temp2 = "";
                if (temptData.attribute(index).isNominal()) {//Can only use stringValue if attribute is categorical
                    temp2 = entry.stringValue(index);
                }

                if (temp.isNaN() || temp2.equals(missingValueSymbol)) {//only care if there is a missing value
                    if (temptData.attribute(index).isNominal()) {//Is discrete/categorical

                        switch (attributeStrategy[index]) {
                            case "mode":
                                entry.setValue(index, attributeMode[index]);
                                break;
                            case "new label":
                                entry.setValue(index, missingValueSymbol);//Due to weka's api, I need to use a value that is already present in attribute 
                                //unless I design weka's dataset manually and include an option for another attribute. So instead just going
                                // to use missingValueSymbol since its already present and I just need a label.
                                break;
                            default:
                                System.out.println("Error in handleMissingValues nominal, did not put in a valid strategy for attribute " + (index + 1));
                        }

                    } else {//is numerical
                        switch (attributeStrategy[index]) {
                            case "mean":
                                entry.setValue(index, attributeMeans[index]);
                                break;
                            case "median":
                                entry.setValue(index, attributeMedian[index]);
                                break;
                            default:
                                System.out.println("Error in handleMissingValues numerical, did not put in a valid strategy for attribute " + (index + 1));
                        }
                    }
                }
            }
        }

        //Updating values for theData.
        attributeMeans = findAllAttributesMeans();
        attributeMode = findAllAttributesModes();
        attributeMedian = findAllAttributesMedians();
    }

    public void constructClassValuesCounter() {
        classValuesCounter = new HashMap<String, Integer>();//Allows quick lookup.
        theData.setClassIndex(classLabelIndex);//prevents this attribute from appearing when enumerate attributes!
        if (theData.attribute(classLabelIndex).isNominal()) {//is a categorical attribute
            Enumeration classValues = theData.attribute(classLabelIndex).enumerateValues();
            while (classValues.hasMoreElements()) {
                classValuesCounter.put(classValues.nextElement().toString(), 0);
            }
        } else {//is a numerical attribute
            System.out.println("Error in constructClassValuesCounter, class label has to be categorical");
        }
    }

    //Call this after cleaning data of missing values
    public void constructValuesCounter() {
        dataValuesCounter = new HashMap<String, Map<String, Map<String, Integer>>>();//top layer is different class labels
        Enumeration classValues = theData.attribute(classLabelIndex).enumerateValues();
        while (classValues.hasMoreElements()) {
            //second layer is attributes minus the class label
            Enumeration attributeNames = theData.enumerateAttributes();

            Map<String, Map<String, Integer>> attributes = new HashMap<String, Map<String, Integer>>();//so column names
            while (attributeNames.hasMoreElements()) {

                Attribute tempAttribute = (Attribute) attributeNames.nextElement();
                String attributeName = tempAttribute.name();//column name

                //third layer is all the class values that appear in an attribute
                Map<String, Integer> valueCounter = new HashMap<String, Integer>();//so values that appear in attribute

                if (theData.attribute(attributeName).isNominal()) {//is categorical so can enumerate through it
                    Enumeration attributeValues = theData.attribute(attributeName).enumerateValues();//enumerating through all values that attribute has (doesn't work for continuous)

                    while (attributeValues.hasMoreElements()) {
                        String temporary = attributeValues.nextElement().toString();
                        valueCounter.put(temporary, 0);
                    }
                }//if attribute is numerical, will add values either in bin approach or gausian approach

                attributes.put(attributeName, valueCounter);
            }
            dataValuesCounter.put(classValues.nextElement().toString(), attributes);
        }
    }

    //calculate the mean of all values in numeric attribute
    private double calculateMean(int atrIndex) {
        double sum = 0.0;
        int numValues = 0;
        if (theData.attribute(atrIndex).isNumeric()) {
            Enumeration entries = theData.enumerateInstances();//enumerating through every entry in theData
            while (entries.hasMoreElements()) {
                Instance anEntry = (Instance) entries.nextElement();
                Double value = anEntry.value(atrIndex);//IF there is no value, this produces NaN
                if (value.isNaN()) {//encountered a missing entry
                    continue;
                } else {
                    numValues++;
                    sum += value;
                }
            }

        } else {//isn't a numerical attribute so default is 0.0
            return 0.0;
        }
        if (numValues == 0) {
            return 0.0;//to avoid dividing by 0. Only occurs if all values are NaN (or attribute is categorical but handled that earlier
        }
        return Math.round((sum / numValues) * 100.0) / 100.0;//rounds it to 2 decimal places.
    }

     //calculate the mode of categorical attribute
    private String calculateStringMode(int atrIndex) {

        Map<String, Integer> valueCounter = new HashMap<String, Integer>();

        if (theData.attribute(atrIndex).isNominal()) {
            Enumeration entries = theData.enumerateInstances();//enumerating through every entry in theData
            while (entries.hasMoreElements()) {
                Instance anEntry = (Instance) entries.nextElement();
                Double temp = anEntry.value(atrIndex);
                if (temp.isNaN()) {//We have a misisng value (i.e. nothing is there)
                    continue;
                }
                String value = anEntry.stringValue(atrIndex);
                if (value.equals(missingValueSymbol)) {//We have a marked missing value.
                    continue;
                }

                valueCounter.merge(value, 1, (a, b) -> a + b);
            }

            String maxKey = Collections.max(valueCounter.entrySet(), Map.Entry.comparingByValue()).getKey();//gets the key with the highest value.
            //In other words, the value that occurs the most in the attribute in given data set

            return maxKey;

        } else {//isn't a discrete/nominal/categorical attribute so default is ""
            return "";
        }
    }

     //calculate the median of numeric attribute
    private double findMedian(int attrIndex) {

        if (theData.attribute(attrIndex).isNominal()) {//Method can't handle analyzing categorical/nominal attributes
            return 0.0;
        }
        Double aValue = theData.kthSmallestValue(attrIndex, theData.size() / 2);

        return aValue;
    }


    //Method to call when want to setup continuous attribute strategies AND start binning if have an attribute want to bin
    public void handleContinuousAttributes(String[] continuousStrategies, Instances tempInstances) {
        prepareGaussian();//In case Gaussian is done.
        contAtrStrategy = continuousStrategies;
        if (continuousStrategies.length != tempInstances.numAttributes()) {
            System.out.println("Error in handleContinuousAttributes. Need widths array to be size of number of attributes including class model if present");
        }
        for (int index = 0; index < continuousStrategies.length; index++) {
            if (isNumeric(continuousStrategies[index])) {
                binNumericalAttribute(index, Double.parseDouble(continuousStrategies[index]), tempInstances);

            } else {
                continue;//Is either Categoriccal or want to do Gause on attribute
            }
        }

    }

    private void binNumericalAttribute(int attributeIndex, double binWidth, Instances tempInstances) {
        if (tempInstances.attribute(attributeIndex).isNumeric()) {
            if (attributeIndex == classLabelIndex || binWidth <= 0) {
                return;
            }
            Enumeration entries = tempInstances.enumerateInstances();//enumerating through every entry in tempInstances
            while (entries.hasMoreElements()) {
                Instance entry = (Instance) entries.nextElement();

                double aValue = entry.value(attributeIndex);
                int counter = 1;
                while (aValue >= counter * binWidth) {//tableau starts binning at 0.
                    counter++;
                }
                entry.setValue(attributeIndex, (counter - 1) * binWidth);//binned value             
            }
        }
    }

    //Method for setting up numeric attribtue variances so when calculate gaussian, don't need to redo calculation over and over.
    private void prepareGaussian() {
        attributeVariances = new double[theData.numAttributes()];
        for (int index = 0; index < theData.numAttributes(); index++) {
            if (index == classLabelIndex) {//not doing gaussian for class label
                continue;
            }
            if (theData.attribute(index).isNumeric()) {
                attributeVariances[index] = theData.variance(index);
            } else {
                attributeVariances[index] = 0;//don't do gaussian for categorical attributes
            }
        }
    }

    //warning if try doing gaussianDistribution for large values, will end up with a gausDistribution of 0 (like if do it for fnlwgt)
    private double gaussianDistribution(int attributeIndex, double value) {
        if (theData.attribute(attributeIndex).isNumeric()) {
            double attributesVariance = attributeVariances[attributeIndex];
            double stdDeviation = Math.sqrt(attributesVariance);

            double gausDistribution = 1.0 / (Math.sqrt(2 * Math.PI) * stdDeviation);
            double exponent = -(Math.pow(value - attributeMeans[attributeIndex], 2));
            gausDistribution *= Math.exp(exponent);

            return gausDistribution;

        } else {
            return 0.0;
        }

    }

    public void constructModel() {
        Enumeration entries = theData.enumerateInstances();//enumerating through every entry in theData
        while (entries.hasMoreElements()) {
            Instance entry = (Instance) entries.nextElement();

            String classLabel;
            if (theData.attribute(classLabelIndex).isNominal()) {
                classLabel = entry.stringValue(classLabelIndex);
            } else {
                System.out.println("Error in constructModel, class label has to be categorical");
                break;
            }

            classValuesCounter.merge(classLabel, 1, (a, b) -> a + b);//incrementing occurances of a specific class label

            for (int index = 0; index < theData.numAttributes(); index++) {
                if (index == classLabelIndex) {
                    continue;//already handled class label.
                }
                if (contAtrStrategy[index].equals("Gause") && theData.attribute(index).isNumeric()) {
                    continue;//Gaussian Distribution was chosen for this continuous/numeric attribute so no reason to count.
                }

                String attributeName = theData.attribute(index).name();
                String entryValue;

                if (theData.attribute(index).isNominal()) {
                    entryValue = entry.stringValue(index);
                } else {
                    entryValue = String.valueOf(entry.value(index));
                }

                dataValuesCounter.get(classLabel).get(attributeName).merge(entryValue, 1, (a, b) -> a + b);
                //THIS is where we increment for categorical AND insert numerical attribtue value and increment it
                // since we had to decide either between binning or gaussian for handling continous

                //1st level is class label  
                //2nd level is the attributes
                //3rd level is the values of attributes
            }
        }
    }

    private boolean isNumeric(String str) {
        try {
            Double.parseDouble(str);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    public List<String> testModel(Instances testDataSet, String[] missingValueStrategies) {//, double[] attributesBinWidth) {
        List<String> predictions = new ArrayList<String>();

        Instances instanceHolder = new Instances(testDataSet);

        handleMissingValues(missingValueStrategies, instanceHolder);//since can't assume testData is clean

        handleContinuousAttributes(contAtrStrategy, instanceHolder);//Have to bin the values like I did with training data

        Enumeration testData = instanceHolder.enumerateInstances();//enumerating through every entry in instanceHolder
        while (testData.hasMoreElements()) {
            Instance anInstance = (Instance) testData.nextElement();

            String[] labelName = new String[classValuesCounter.size()];
            Double[] labelProbability = new Double[classValuesCounter.size()];
            int counter = 0;//so know what index we are at for above two arrays

            Enumeration classValues = theData.attribute(classLabelIndex).enumerateValues();
            //calculating the probability of P(x|c)
            while (classValues.hasMoreElements()) {//we iterate through possible class values entry could be
                double probabilityTemp = 1.0;
                String classValue = (String) classValues.nextElement();
                int classValueCount = classValuesCounter.get(classValue);

                for (int attributeNum = 0; attributeNum < anInstance.numAttributes() - 1; attributeNum++) {
                    String attributeName = theData.attribute(attributeNum).name();

                    String valueAtInstance;
                    if (instanceHolder.attribute(attributeNum).isNominal()) {
                        valueAtInstance = anInstance.stringValue(attributeNum);
                    } else {
                        valueAtInstance = String.valueOf(anInstance.value(attributeNum));
                    }


                    if (contAtrStrategy[attributeNum].equals("Gause") && theData.attribute(attributeNum).isNumeric()) {//Gaussian distribution was chosen for this continuous attribute over binning
                        probabilityTemp *= gaussianDistribution(attributeNum, Double.parseDouble(valueAtInstance));
                    } else {

                        /*
                     Doing +1 to handle 0 probabiltiy would require keeping a seperate class label count for each attribute since
                     numbers would be different due to different amount of values in each attribute. So instead of creating several
                     collections just to keep track of adjusted probabilities due to 0 probabiltiy handling, just going to do
                     1/(classValueCount+1) for any value that has 0 for counter OR did not appear in training data set.
                     Not the most mathmatically sound but neither is doing +1 when having several different attributes that would require
                     either different class label counters for each OR increase counters for values in attributes by more than 1 in order
                     to match up with next class vlaue counter (which would be double its original)
                         */
                        if (dataValuesCounter.get(classValue).get(attributeName).containsKey(valueAtInstance)) {
                            double countOfValue = dataValuesCounter.get(classValue).get(attributeName).get(valueAtInstance);
                            if (countOfValue == 0.0) {
                                probabilityTemp *= (1.0 / (classValueCount + 1.0));
                            } else {
                                probabilityTemp *= (countOfValue / classValueCount);
                            }
                        } else {

                            probabilityTemp *= (1.0 / (classValueCount + 1.0));

                        }
                    }

                }
                labelName[counter] = classValue;
                labelProbability[counter] = probabilityTemp * (classValueCount / (theData.numInstances() * 1.0));//P(X|c) * P(c)
                counter++;

            }

            int highestValueIndex = 0;
            for (int i = 1; i < labelProbability.length; i++) {
                if (labelProbability[highestValueIndex] < labelProbability[i]) {
                    highestValueIndex = i;
                }
            }

            predictions.add(labelName[highestValueIndex]);

        }
        return predictions;
    }

}
