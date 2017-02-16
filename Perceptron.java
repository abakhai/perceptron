/**
 *  The Perceptron Algorithm
 *  By Dr Noureddin Sadawi
 *  (Adapted by Mayank Gupta)
 *  Please watch my Youtube videos on perceptron for things to make sense! : https://youtu.be/4aksMtJHWEQ
 *  Copyright (C) 2014 
 *  @author Dr Noureddin Sadawi 
 *  
 *  This program is free software: you can redistribute it and/or modify
 *  it as you wish ONLY for legal and ethical purposes
 * 
 *  I ask you only, as a professional courtesy, to cite my name, web page 
 *  and my YouTube Channel!
 *  
 *    Code adapted from:
 *    https://github.com/RichardKnop/ansi-c-perceptron
 */  

import java.text.*;

class Perceptron
{
	static int MAX_ITER = 100;
	static double LEARNING_RATE = 0.1;           
	static int NUM_INSTANCES = 100;
	static int theta = 0;  
	public static void main(String args[]){ 
		//three variables (features)                      
		double[] x1 = new double [NUM_INSTANCES];    
		double[] x2 = new double [NUM_INSTANCES];
		double[] x3 = new double [NUM_INSTANCES];
		int[] outputs = new int [NUM_INSTANCES];

		//fifty random points of class 1
		for(int i = 0; i < NUM_INSTANCES/2; i++){
			x1[i] = randomNumber(5 , 10);
			x2[i] = randomNumber(4 , 8); 
			x3[i] = randomNumber(2 , 9);        
			outputs[i] = 1;         
			System.out.println(x1[i]+"\t"+x2[i]+"\t"+x3[i]+"\t"+outputs[i]);
		}

		//fifty random points of class 0
		for(int i = 50; i < NUM_INSTANCES; i++){
			x1[i] = randomNumber(-1 , 3);
			x2[i] = randomNumber(-4 , 2);   
			x3[i] = randomNumber(-3 , 5);       
			outputs[i] = 0;        
			System.out.println(x1[i]+"\t"+x2[i]+"\t"+x3[i]+"\t"+outputs[i]);
		}

		double[] weights = new double[4];// 3 for input variables and one for bias
		double localError, globalError;
		int iteration, output;

		weights[0] = randomNumber(0,1);// w1
		weights[1] = randomNumber(0,1);// w2
		weights[2] = randomNumber(0,1);// w3
		weights[3] = randomNumber(0,1);// this is the bias

		iteration = 0;
		do {
			iteration++;
			globalError = 0;
			//loop through all instances (complete one epoch)
			for (int p = 0; p < NUM_INSTANCES; p++) {
				// calculate predicted class
				output = calculateOutput(theta,weights, x1[p], x2[p], x3[p]);
				// difference between predicted and actual class values
				localError = outputs[p] - output;
				//update weights and bias
				weights[0] += LEARNING_RATE * localError * x1[p];
				weights[1] += LEARNING_RATE * localError * x2[p];
				weights[2] += LEARNING_RATE * localError * x3[p];
				weights[3] += LEARNING_RATE * localError;
				//summation of squared error (error value for all instances)
				globalError += (localError*localError);
			}

			/* Root Mean Squared Error */
			System.out.println("Iteration "+iteration+" : RMSE = "+Math.sqrt(globalError/NUM_INSTANCES));
		} while (globalError != 0 && iteration<=MAX_ITER);

		System.out.println("\n=======\nDecision boundary equation:");
		System.out.println(weights[0] +"*x + "+weights[1]+"*x2 +  "+weights[2]+"*x3 + "+weights[3]+" = 0");

		//generate 10 new random points and check their classes
		//notice the range of -10 and 10 means the new point could be of class 1 or 0
		//-10 to 10 covers all the ranges we used in generating the 50 classes of 1's and 0's above
		for(int j = 0; j < 10; j++){
			double x4 = randomNumber(-10 , 10);
			double y4 = randomNumber(-10 , 10);   
			double z4 = randomNumber(-10 , 10); 

			output = calculateOutput(theta,weights, x4, y4, z4);
			System.out.println("\n=======\nNew Random Point:");
			System.out.println("x = "+x4+",y = "+y4+ ",z = "+z4);
			System.out.println("class = "+output);
		}
	}//end main  

	/**
	 * returns a random double value within a given range
	 * @param min the minimum value of the required range (int)
	 * @param max the maximum value of the required range (int)
	 * @return a random double value between min and max
	 */ 
	public static double randomNumber(int min , int max) {
		DecimalFormat df = new DecimalFormat("#.####");
		double d = min + Math.random() * (max - min);
		String s = df.format(d);
		double x = Double.parseDouble(s);
		return x;
	}

	/**
	 * returns either 1 or 0 using a threshold function
	 * theta is 0range
	 * @param theta an integer value for the threshold
	 * @param weights[] the array of weights
	 * @param x the x input value
	 * @param x2 the x2 input value
	 * @param x3 the x3 input value
	 * @return 1 or 0
	 */ 
	static int calculateOutput(int theta, double weights[], double x1, double x2, double x3)
	{
		double sum = x1 * weights[0] + x2 * weights[1] + x3 * weights[2] + weights[3];
		return (sum >= theta) ? 1 : 0;
	}

}
