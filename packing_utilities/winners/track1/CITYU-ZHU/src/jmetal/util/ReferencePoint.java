package jmetal.util;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.StringTokenizer;
import java.util.Vector;

import jmetal.core.SolutionSet;
import jmetal.qualityIndicator.util.MetricsUtil;

/**
 * @author Rubén Saborido Infantes
 * This class offers different methods to manipulate reference points.
 * A reference point is a vector containing a value for each component of an objective function.
 */
public class ReferencePoint {

	Vector<Double> referencePoint_;

	public enum ReferencePointType {
		ACHIEVABLE, UNACHIEVABLE
	};

	/**
	 * Construct a reference point reading it from a file
	 * @param referencePointFileName File containing a reference point	 
	 */
	public ReferencePoint(String referencePointFileName) throws IOException {
		// Open the aspiration file
		FileInputStream fis = new FileInputStream(referencePointFileName);
		InputStreamReader isr = new InputStreamReader(fis);
		BufferedReader br = new BufferedReader(isr);

		referencePoint_ = new Vector<Double>();
		String aux = br.readLine();
		while (aux != null) {
			StringTokenizer st = new StringTokenizer(aux);

			while (st.hasMoreTokens()) {
				Double value = (new Double(st.nextToken()));
				referencePoint_.add(value);
			}
			aux = br.readLine();
		}
		br.close();
	}

	/**
	 * Construct a reference point from a vector
	 * @param referencePoint Vector defining a reference point	 	 
	 */
	public ReferencePoint(double[] referencePoint) {
		this.referencePoint_ = new Vector<Double>();

		for (int indexOfComponent = 0; indexOfComponent < referencePoint.length; indexOfComponent++)
			this.referencePoint_.add(Double
					.valueOf(referencePoint[indexOfComponent]));
	}

	/**
	 * Construct an empty reference point with a dimension given
	 * @param numberOfObjectives The number of components 
	 */
	public ReferencePoint(int numberOfObjectives) {
		this.referencePoint_ = new Vector<Double>(numberOfObjectives);
		referencePoint_.setSize(numberOfObjectives);
	}

	/**
	 * Construct a random reference point from a Pareto front file
	 * @param type The type of the created reference point
	 * @param paretoFrontFileName A Pareto front in a file
	 */
	public ReferencePoint(ReferencePointType type, String paretoFrontFileName)
			throws JMException {
		int randomIndexPoint;
		double[] minimumValues, maximumValues;
		double[][] front;
		int index, numberOfObjectives;
		MetricsUtil metrics = new MetricsUtil();

		front = metrics.readFront(paretoFrontFileName);

		numberOfObjectives = front[0].length;

		minimumValues = metrics.getMinimumValues(front, numberOfObjectives);
		maximumValues = metrics.getMaximumValues(front, numberOfObjectives);

		randomIndexPoint = PseudoRandom.randInt(0, front.length);

		referencePoint_ = new Vector<Double>();

		switch (type) {
		case ACHIEVABLE:
			for (index = 0; index < numberOfObjectives; index++) {
				this.referencePoint_.add(PseudoRandom.randDouble(
						front[randomIndexPoint][index], maximumValues[index]));
			}
			break;

		case UNACHIEVABLE:
			for (index = 0; index < numberOfObjectives; index++) {
				this.referencePoint_.add(PseudoRandom.randDouble(
						minimumValues[index], front[randomIndexPoint][index]));
			}
			break;
		}
	}
	
	/**
	 * Get a component of the reference point
	 * @param indexOfObjective The index of the component 
	 * @return The value of the selected component
	 */
	public Double get(int indexOfObjective) {
		return this.referencePoint_.get(indexOfObjective);
	}

	/**
	 * Set the value of a component of the reference point
	 * @param indexOfObjective The index of the component
	 * @param valueOfObjective The new value of the component 
	 * @return The value of the selected component
	 */
	public void set(int indexOfObjective, Double valueOfObjective) {
		this.referencePoint_.set(indexOfObjective, valueOfObjective);
	}

	/**
	 * Get the size of the reference point
	 * @return The number of components of the reference point
	 */
	public int size() {
		return this.referencePoint_.size();
	}

	/**
	 * Convert the reference point in a vector of double
	 * @return A vector of double containing the values of the reference point
	 */
	public double[] toDouble() {
		double[] result = new double[this.referencePoint_.size()];
		for (int indexOfObjective = 0; indexOfObjective < this.referencePoint_
				.size(); indexOfObjective++) {
			result[indexOfObjective] = referencePoint_.get(indexOfObjective)
					.doubleValue();
		}

		return result;
	}

	/**
	 * Return the solutions Pareto-dominated by the reference point
	 * @param solutions A set of solutions	
	 * @return The solutions Pareto-dominated by the reference point
	 */
	public double[][] getDominatedSolutionsByMe(double[][] solutions) {
		double[][] result;
		ArrayList<Integer> indexsOfDominatedSolutions = new ArrayList<Integer>();

		for (int indexOfSolution = 0; indexOfSolution < solutions.length; indexOfSolution++) {
			if (ParetoDominance.checkParetoDominance(this.toDouble(),
					solutions[indexOfSolution]) == -1) {
				indexsOfDominatedSolutions
						.add(Integer.valueOf(indexOfSolution));
			}
		}

		result = new double[indexsOfDominatedSolutions.size()][referencePoint_
				.size()];
		for (int indexOfSolution = 0; indexOfSolution < indexsOfDominatedSolutions
				.size(); indexOfSolution++) {
			result[indexOfSolution] = solutions[indexsOfDominatedSolutions
					.get(indexOfSolution)].clone();
		}

		return result;
	}

	/**
	 * Return the solutions which Pareto-dominate to the reference point
	 * @param solutions A set of solutions	
	 * @return The solutions which Pareto-dominate to the reference point
	 */
	public double[][] getDominantSolutions(double[][] solutions) {
		double[][] result;
		ArrayList<Integer> indexsOfDominatedSolutions = new ArrayList<Integer>();

		for (int indexOfSolution = 0; indexOfSolution < solutions.length; indexOfSolution++) {
			if (ParetoDominance.checkParetoDominance(this.toDouble(),
					solutions[indexOfSolution]) == 1) {
				indexsOfDominatedSolutions
						.add(Integer.valueOf(indexOfSolution));
			}
		}

		result = new double[indexsOfDominatedSolutions.size()][referencePoint_
				.size()];
		for (int indexOfSolution = 0; indexOfSolution < indexsOfDominatedSolutions
				.size(); indexOfSolution++) {
			result[indexOfSolution] = solutions[indexsOfDominatedSolutions
					.get(indexOfSolution)].clone();
		}

		return result;
	}

	/**
	 * Return the solutions greater of equal than the reference point
	 * @param solutions A set of solutions	
	 * @return The solutions greater of equal than the reference point
	 */
	public double[][] getSolutionsGreaterOrEqualThanMe(double[][] solutions) {
		double[][] result;

		ArrayList<Integer> indexsOfSolutions = new ArrayList<Integer>();

		for (int indexOfSolution = 0; indexOfSolution < solutions.length; indexOfSolution++) {
			boolean isGreater = true;
			int indexOfObjective = 0;

			while (isGreater
					&& indexOfObjective < solutions[indexOfSolution].length) {
				isGreater = solutions[indexOfSolution][indexOfObjective] >= this.referencePoint_
						.get(indexOfObjective);
				indexOfObjective++;
			}

			if (isGreater) {
				indexsOfSolutions.add(Integer.valueOf(indexOfSolution));
			}
		}

		result = new double[indexsOfSolutions.size()][referencePoint_.size()];
		for (int indexOfSolution = 0; indexOfSolution < indexsOfSolutions
				.size(); indexOfSolution++) {
			result[indexOfSolution] = solutions[indexsOfSolutions
					.get(indexOfSolution)].clone();
		}

		return result;
	}

	/**
	 * Return the solutions greater of equal than the reference point
	 * @param solutions A set of solutions	
	 * @return The solutions greater of equal than the reference point
	 */
	public SolutionSet getSolutionsGreaterOrEqualThanMe(SolutionSet solutions) {
		ArrayList<Integer> indexsOfSolutions = new ArrayList<Integer>();

		for (int indexOfSolution = 0; indexOfSolution < solutions.size(); indexOfSolution++) {
			boolean isGreater = true;
			int indexOfObjective = 0;

			while (isGreater
					&& indexOfObjective < solutions.get(indexOfSolution)
							.getNumberOfObjectives()) {
				isGreater = solutions.get(indexOfSolution).getObjective(
						indexOfObjective) >= this.referencePoint_
						.get(indexOfObjective);
				indexOfObjective++;
			}

			if (isGreater) {
				indexsOfSolutions.add(Integer.valueOf(indexOfSolution));		
			}
		}

		SolutionSet result = new SolutionSet(indexsOfSolutions.size());
		for (int indexOfSolution = 0; indexOfSolution < indexsOfSolutions.size(); indexOfSolution++)
		{
			result.add(solutions.get(indexsOfSolutions.get(indexOfSolution)));
		}

		return result;
	}

	/**
	 * Return the solutions lower of equal than the reference point
	 * @param solutions A set of solutions	
	 * @return The solutions lower of equal than the reference point
	 */
	public double[][] getSolutionsLowerOrEqualThanMe(double[][] solutions) {
		double[][] result;

		ArrayList<Integer> indexsOfSolutions = new ArrayList<Integer>();

		for (int indexOfSolution = 0; indexOfSolution < solutions.length; indexOfSolution++) {
			boolean isLower = true;
			int indexOfObjective = 0;

			while (isLower
					&& indexOfObjective < solutions[indexOfSolution].length) {
				isLower = solutions[indexOfSolution][indexOfObjective] <= this.referencePoint_
						.get(indexOfObjective);
				indexOfObjective++;
			}

			if (isLower) {
				indexsOfSolutions.add(Integer.valueOf(indexOfSolution));
			}
		}

		result = new double[indexsOfSolutions.size()][referencePoint_.size()];
		for (int indexOfSolution = 0; indexOfSolution < indexsOfSolutions
				.size(); indexOfSolution++) {
			result[indexOfSolution] = solutions[indexsOfSolutions
					.get(indexOfSolution)].clone();
		}

		return result;
	}

	/**
	 * Return the solutions lower of equal than the reference point
	 * @param solutions A set of solutions	
	 * @return The solutions lower of equal than the reference point
	 */
	public SolutionSet getSolutionsLowerOrEqualThanMe(SolutionSet solutions) {		
		ArrayList<Integer> indexsOfSolutions = new ArrayList<Integer>();
		
		for (int indexOfSolution = 0; indexOfSolution < solutions.size(); indexOfSolution++) {
			boolean isLower = true;
			int indexOfObjective = 0;

			while (isLower
					&& indexOfObjective < solutions.get(indexOfSolution).getNumberOfObjectives()) {
				isLower = solutions.get(indexOfSolution).getObjective(indexOfObjective) <= this.referencePoint_
						.get(indexOfObjective);
				indexOfObjective++;
			}

			if (isLower) {
				indexsOfSolutions.add(Integer.valueOf(indexOfSolution));		
			}
		}

		SolutionSet result = new SolutionSet(indexsOfSolutions.size());
		for (int indexOfSolution = 0; indexOfSolution < indexsOfSolutions.size(); indexOfSolution++)
		{
			result.add(solutions.get(indexsOfSolutions.get(indexOfSolution)));
		}
		
		return result;
	}	
}
