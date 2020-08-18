/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Executor;

/**
 *
 * @author Matthew Berkman
 */
import MiniProject1.ModelEvaluation;


public class MainClass {

    public static final String CSVFilePath = "C:\\Users\\MatthewB\\Documents\\DataMiniProject\\TheData.csv";

    public static void main(String[] args) throws Exception {

        ModelEvaluation evaluationModel = new ModelEvaluation(CSVFilePath);

        String[] strategies = {"mean", "new label", "mean", "mode", "mean", "mode", "new label", "mode", "mode", "mode", "median", "mean", "mean", "mode", ""};

        String[] continuousAtributeStrategies = {"3","Category","300","Category","1","Category","Category","Category","Category","Category","5000","270","5","Category","Category"};
        
        evaluationModel.setMissingValueStrategies(strategies);
        //setting negative value for a binWidth means want to use Gaussian for that attribute
        evaluationModel.setContinuousAttributeStrategies(continuousAtributeStrategies);
        
        evaluationModel.performKFoldValidation(10);
        
        System.out.println("Models accuracy " + evaluationModel.getModelsAccuracy());
        System.out.println("===================");
        System.out.println("For Class Label " + evaluationModel.getPositiveClassLabel());
        System.out.println(" Precision " + evaluationModel.getPositiveLabelPrecision());
        System.out.println(" Recall " + evaluationModel.getPositiveLabelRecall());
        System.out.println(" F1Measure " + evaluationModel.getPositiveLabelF1());
        System.out.println("===================");
        System.out.println("For Class Label " + evaluationModel.getNegativeClassLabel());
        System.out.println(" Precision " + evaluationModel.getNegativeLabelPrecision());
        System.out.println(" Recall " + evaluationModel.getNegativeLabelRecall());
        System.out.println(" F1Measure " + evaluationModel.getNegativeLabelF1());
        
    }
}
