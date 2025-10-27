package weka.classifiers.trees;

import weka.core.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import javax.swing.*;
import java.awt.*;

/**
 * A splitter for using the best infogain-based bivariate split on numeric attributes.
 */
public class BivariateNumericAttributeSplitter extends Splitter {
    
    /** The natural logarithm of 2 */
    public static final double log2 = Math.log(2);

    /** The first attribute that will be used for splitting */
    private Attribute attX1;

    /** The second attribute that will be used for splitting */
    private Attribute attX2;
    
    /** The first coordinate for the split */
    private double x1;
    
    /** The second coordinate for the split */
    private double x2;
    
    /** The radius for the split */
    private double r;

    /** The split quality: the information gain */
    private double splitQuality;
    
    /**
     * Finding the best bivariate split across attributes using information gain.
     */
    public void findSplit(Instances data) {
		int[] initialCounts = new int[data.numClasses()];
			for(Instance instance : data){
				initialCounts[(int)instance.classValue()]++;
			}
			//For every combination of attributes find the best split.
			//Little brute-forcy but it works
			for (Attribute a1 : Collections.list(data.enumerateAttributes())){
				for (Attribute a2 : Collections.list(data.enumerateAttributes())){
					if(a1 != a2){
						findBestSplit(data, a1, a2, initialCounts);
					}
				}
			}
	    }

    /**
     * Decides whether the given instance belongs to the first branch or not.
     */
    public boolean firstSubset(Instance instance) {
		boolean inTheCircle = ((instance.value(attX1) - x1) * (instance.value(attX1) - x1)
		 + (instance.value(attX2) - x2) * (instance.value(attX2) - x2) < r);

		return inTheCircle;
    }

    /**
     * Returns the information gain of the best split found.
     */
    public double splitQuality() {
	return splitQuality;
    }

    /**
     * Returns a string representation of the condition for a branch.
     */
    public String toString(boolean leftBranch, int decimalPlaces) {
        return "Attributes: " + attX1.name() + "," + attX2.name() +
	    (leftBranch ? " inside " : " outside ") + "circle with center (" +
	    String.format("%." + decimalPlaces + "f", x1) + "," +
	    String.format("%." + decimalPlaces + "f", x2) + ") and radius " +
	    String.format("%." + decimalPlaces + "f", r); 
    }
    
    /**
     * Plot the split, along with a scatterplot of the relevant data.
     */
    public void plot(Instances data, int decimalPlaces) {
	Color[] colours = { Color.blue, Color.red, Color.green,
	    Color.cyan, Color.pink, new Color(255, 0, 255), Color.orange,
	    new Color(255, 0, 0), new Color(0, 255, 0), Color.white };
	int padding = 50;
	double minX = data.stream().mapToDouble(p -> p.value(attX1)).min().getAsDouble();
	double maxX = data.stream().mapToDouble(p -> p.value(attX1)).max().getAsDouble();
	double minY = data.stream().mapToDouble(p -> p.value(attX2)).min().getAsDouble();
	double maxY = data.stream().mapToDouble(p -> p.value(attX2)).max().getAsDouble();
	JFrame jf = new JFrame("X1: " + attX1.name() + " Range: " +
			       String.format("%." + decimalPlaces + "f", minX) + "," +
			       String.format("%." + decimalPlaces + "f", maxX) +
			       " X2: " + attX2.name() + " Range: " +
			       String.format("%." + decimalPlaces + "f", minY) + "," +
			       String.format("%." + decimalPlaces + "f", maxY));
	jf.add(new JPanel() {
		private int getX(double v) {
		    return (int) ((v - minX) / (maxX - minX) * (getWidth() - 2 * padding)) + padding;
		}
		private int getY(double v) {
		    return (int) ((maxY - v) / (maxY - minY) * (getHeight() - 2 * padding)) + padding;
		}
		private int getXradius(double v) {
		    return (int) (v / (maxX - minX) * (getWidth() - 2 * padding));
		}
		private int getYradius(double v) {
		    return (int) (v / (maxY - minY) * (getHeight() - 2 * padding));
		}
		protected void paintComponent(Graphics g) {
		    super.paintComponent(g);
		    for (Instance i : data) {
			g.setColor(colours[((int)i.classValue()) % colours.length]);
			g.fillOval(getX(i.value(attX1)) - 4, getY(i.value(attX2)) - 4, 8, 8);
		    }
		    g.drawOval(getX(x1) - getXradius(r), getY(x2) - getYradius(r), 2 * getXradius(r), 2 * getYradius(r));
		}
	    });
	jf.setSize(600, 600);
	jf.setVisible(true);
	JOptionPane pane = new JOptionPane(null);
	JDialog d = pane.createDialog(jf, null);
	d.setLocationRelativeTo(jf);
	d.setLocation(0,0);
	d.setSize(new Dimension(10,30));
	d.setVisible(true);
	jf.dispose();
    }
    
    /**
     * Finds the best circular split based on the given reduced data.
     */
    private void findBestSplit(Instances data, Attribute a1, Attribute a2, int[] initialCounts) {
		int[] countsLeft = Arrays.copyOf(initialCounts, data.numClasses());
		int[] countsRight = new int[data.numClasses()];

		//Previous Values
		double prevX = Double.NEGATIVE_INFINITY;
		double prevY = Double.NEGATIVE_INFINITY;
		
		data.sort(a1);

		/*For each instance 
		  Draw a distance to the next point (set radius to 1/2 of the distance to the next point)
		  NOTE: For looks 1 ahead. 
		*/
		for(Instance instance : data)
		{
			// If (distance between the center of the circle and the current point) > radius
			if(Math.sqrt((instance.value(a1) - x1 * instance.value(a1) - x1) + (instance.value(a2) - x2 * instance.value(a2) - x2)) > r)
			{
				double currentSplitQuality = informationGain(countsLeft, countsRight);
				if(currentSplitQuality > splitQuality)
				{
					splitQuality = currentSplitQuality;
					
					// x1 =  The first coordinate for the split
					// x2 =  The second coordinate for the split == y
					// r  =  Radius

					x1 = prevX;
					x2 = prevY;

					attX1 = a1;
					attX2 = a2;
					
					//a^2 + b^2 = c^2
					double distance = (instance.value(a1) - prevX) * (instance.value(a1) - prevX) + (instance.value(a2) - prevY) * (instance.value(a2) - prevY);
					//r = r_{max}/r_{min}/2  
					r = Math.sqrt(distance)/2;
				}
				prevX = instance.value(a1);
				prevY = instance.value(a2);
			}
			countsLeft[(int)instance.classValue()]--;
			countsRight[(int)instance.classValue()]++;
		}
    }
    
    /**
     * Helper method for computing entropy.
     */
    public static double lnFunc(int num){
        return (num <= 0) ? 0 : num * Math.log(num);
    }
    
    /**
     * Computes the base-2 information gain for the given array of counts.
     */
    public static double informationGain(int[] leftCounts, int[] rightCounts) {
	double sumOfNLogNLeft = 0, sumOfNLogNRight = 0, sumOfNLogNTotal = 0;
	int sumLeft = 0, sumRight = 0, sumTotal = 0;
	for (int i = 0; i < leftCounts.length; i++) {
	    sumOfNLogNLeft -= lnFunc(leftCounts[i]);
	    sumLeft += leftCounts[i];
	    sumOfNLogNRight -= lnFunc(rightCounts[i]);
	    sumRight += rightCounts[i];
	    sumOfNLogNTotal -= lnFunc(leftCounts[i] + rightCounts[i]);
	}
	sumTotal = sumLeft + sumRight;
        return (sumTotal <= 0) ? 0 :
	    (sumOfNLogNTotal + lnFunc(sumTotal) -
	     (sumOfNLogNLeft + lnFunc(sumLeft) + sumOfNLogNRight + lnFunc(sumRight))) /
	    (sumTotal * log2);
    }
}
