//  MOEAD_SDVRP 
//  For EMO2021 Huawei VRP competition
//
//  Author:         LIU Fei  
//  E-mail:         fliu36-c@my.cityu.edu.hk
//  Create Date:    2021.2.1
//  Last modified   Date: 2021.2.2
//

package jmetal.operators.crossover;

import jmetal.core.Solution;
import jmetal.encodings.solutionType.ArrayRealAndBinarySolutionType;
import jmetal.encodings.solutionType.PermutationBinarySolutionType;
import jmetal.encodings.variable.Binary;
import jmetal.encodings.variable.Permutation;
import jmetal.util.Configuration;
import jmetal.util.JMException;
import jmetal.util.PseudoRandom;
import jmetal.util.wrapper.XReal;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class PMXSinglePointCrossover extends Crossover {
	/**
	 * EPS defines the minimum difference allowed between real values
	 */
	// private static final double EPS= 1.0e-14;

	private static final double ETA_C_DEFAULT_ = 20.0;
	private Double PMXCrossoverProbability_ = null;
	private Double binaryCrossoverProbability_ = null;
	private double distributionIndex_ = ETA_C_DEFAULT_;

	/**
	 * Valid solution types to apply this operator
	 */
	private static final List VALID_TYPES = Arrays.asList(PermutationBinarySolutionType.class);

	/**
	 * Constructor
	 */
	public PMXSinglePointCrossover(HashMap<String, Object> parameters) {
		super(parameters);

		if (parameters.get("PMXCrossoverProbability") != null)
			PMXCrossoverProbability_ = (Double) parameters.get("PMXCrossoverProbability");
		if (parameters.get("binaryCrossoverProbability") != null)
			binaryCrossoverProbability_ = (Double) parameters.get("binaryCrossoverProbability");
		if (parameters.get("distributionIndex") != null)
			distributionIndex_ = (Double) parameters.get("distributionIndex");
	} // Constructor

	/**
	 * Perform the crossover operation.
	 * 
	 * @param realProbability Crossover probability
	 * @param parent1         The first parent
	 * @param parent2         The second parent
	 * @return An array containing the two offsprings
	 */
	public Solution[] doCrossover(Double PMXProbability, Double binaryProbability, Solution parent1, Solution parent2)
			throws JMException {

		Solution[] offSpring = new Solution[2];

		offSpring[0] = new Solution(parent1);
		offSpring[1] = new Solution(parent2);

		// PMX crossover
		int permutationLength;

		permutationLength = ((Permutation) parent1.getDecisionVariables()[0]).getLength();

		int parent1Vector[] = ((Permutation) parent1.getDecisionVariables()[0]).vector_;
		int parent2Vector[] = ((Permutation) parent2.getDecisionVariables()[0]).vector_;
		int offspring1Vector[] = ((Permutation) offSpring[0].getDecisionVariables()[0]).vector_;
		int offspring2Vector[] = ((Permutation) offSpring[1].getDecisionVariables()[0]).vector_;

		if (PseudoRandom.randDouble() < PMXProbability) {
			int cuttingPoint1;
			int cuttingPoint2;

			// STEP 1: Get two cutting points
			cuttingPoint1 = PseudoRandom.randInt(0, permutationLength - 1);
			cuttingPoint2 = PseudoRandom.randInt(0, permutationLength - 1);
			while (cuttingPoint2 == cuttingPoint1)
				cuttingPoint2 = PseudoRandom.randInt(0, permutationLength - 1);

			if (cuttingPoint1 > cuttingPoint2) {
				int swap;
				swap = cuttingPoint1;
				cuttingPoint1 = cuttingPoint2;
				cuttingPoint2 = swap;
			} // if
				// STEP 2: Get the subchains to interchange
			int replacement1[] = new int[permutationLength];
			int replacement2[] = new int[permutationLength];
			for (int i = 0; i < permutationLength; i++)
				replacement1[i] = replacement2[i] = -1;

			// STEP 3: Interchange
			for (int i = cuttingPoint1; i <= cuttingPoint2; i++) {
				offspring1Vector[i] = parent2Vector[i];
				offspring2Vector[i] = parent1Vector[i];

				replacement1[parent2Vector[i]] = parent1Vector[i];
				replacement2[parent1Vector[i]] = parent2Vector[i];
			} // for

			// STEP 4: Repair offsprings
			for (int i = 0; i < permutationLength; i++) {
				if ((i >= cuttingPoint1) && (i <= cuttingPoint2))
					continue;

				int n1 = parent1Vector[i];
				int m1 = replacement1[n1];

				int n2 = parent2Vector[i];
				int m2 = replacement2[n2];

				while (m1 != -1) {
					n1 = m1;
					m1 = replacement1[m1];
				} // while
				while (m2 != -1) {
					n2 = m2;
					m2 = replacement2[m2];
				} // while
				offspring1Vector[i] = n1;
				offspring2Vector[i] = n2;
			} // for
		} // if

		// Single point crossover

		if (PseudoRandom.randDouble() <= binaryProbability) {
			Binary binaryChild0 = (Binary) offSpring[0].getDecisionVariables()[1];
			Binary binaryChild1 = (Binary) offSpring[1].getDecisionVariables()[1];

			int totalNumberOfBits = binaryChild0.getNumberOfBits();

			// 2. Calcule the point to make the crossover
			int crossoverPoint = PseudoRandom.randInt(0, totalNumberOfBits - 1);

			// 5. Make the crossover;
			for (int i = crossoverPoint; i < totalNumberOfBits; i++) {
				boolean swap = binaryChild0.bits_.get(i);
				binaryChild0.bits_.set(i, binaryChild1.bits_.get(i));
				binaryChild1.bits_.set(i, swap);
			} // for
		} // if

		return offSpring;
	} // doCrossover

	@Override
	public Object execute(Object object) throws JMException {
		Solution[] parents = (Solution[]) object;

		if (parents.length != 2) {
			Configuration.logger_.severe("PMXSinglePointCrossover.execute: operator " + "needs two parents");
			Class cls = java.lang.String.class;
			String name = cls.getName();
			throw new JMException("Exception in " + name + ".execute()");
		} // if

		if (!(VALID_TYPES.contains(parents[0].getType().getClass())
				&& VALID_TYPES.contains(parents[1].getType().getClass()))) {
			Configuration.logger_.severe("PMXSinglePointCrossover.execute: the solutions " + "type "
					+ parents[0].getType() + " is not allowed with this operator");

			Class cls = java.lang.String.class;
			String name = cls.getName();
			throw new JMException("Exception in " + name + ".execute()");
		} // if
		Solution[] offSpring;
		offSpring = doCrossover(PMXCrossoverProbability_, binaryCrossoverProbability_, parents[0], parents[1]);

		return offSpring;
	} // execute

} // SBXSinglePointCrossover
