//  MOEAD_SDVRP 
//  For EMO2021 Huawei VRP competition
//
//  Author:         LIU Fei  
//  E-mail:         fliu36-c@my.cityu.edu.hk
//  Create Date:    2021.2.1
//  Last modified   Date: 2021.2.2
//

package jmetal.operators.mutation;

import jmetal.core.Solution;
import jmetal.encodings.solutionType.ArrayRealAndBinarySolutionType;
import jmetal.encodings.solutionType.PermutationSolutionType;
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

public class SwapBitFlipMutation extends Mutation {
	private static final double ETA_M_DEFAULT_ = 20.0;
	private final double eta_m_ = ETA_M_DEFAULT_;

	private Double SwapMutationProbability_ = null;
	private Double binaryMutationProbability_ = null;
	private double distributionIndex_ = eta_m_;

	/**
	 * Valid solution types to apply this operator
	 */
	private static final List VALID_TYPES = Arrays.asList(PermutationBinarySolutionType.class);

	/**
	 * Constructor
	 */
	public SwapBitFlipMutation(HashMap<String, Object> parameters) {
		super(parameters);
		if (parameters.get("SwapMutationProbability") != null)
			SwapMutationProbability_ = (Double) parameters.get("SwapMutationProbability");
		if (parameters.get("binaryMutationProbability") != null)
			binaryMutationProbability_ = (Double) parameters.get("binaryMutationProbability");
		if (parameters.get("distributionIndex") != null)
			distributionIndex_ = (Double) parameters.get("distributionIndex");
	} // PolynomialBitFlipMutation

	@Override
	public Object execute(Object object) throws JMException {
		Solution solution = (Solution) object;

		if (!VALID_TYPES.contains(solution.getType().getClass())) {
			Configuration.logger_.severe("SwapBitFlipMutation.execute: the solution " + "type " + solution.getType()
					+ " is not allowed with this operator");

			Class cls = java.lang.String.class;
			String name = cls.getName();
			throw new JMException("Exception in " + name + ".execute()");
		} // if

		doMutation(SwapMutationProbability_, binaryMutationProbability_, solution);
		return solution;
	} // execute

	/**
	 * doMutation method
	 * 
	 * @param realProbability
	 * @param binaryProbability
	 * @param solution
	 * @throws JMException
	 */
	public void doMutation(Double SwapProbability, Double binaryProbability, Solution solution) throws JMException {
		double rnd, delta1, delta2, mut_pow, deltaq;
		double y, yl, yu, val, xy;

		// Polynomial mutation applied to the array real
		int permutation[];
		int permutationLength;
		if (solution.getType().getClass() == PermutationBinarySolutionType.class) {

			permutationLength = ((Permutation) solution.getDecisionVariables()[0]).getLength();
			permutation = ((Permutation) solution.getDecisionVariables()[0]).vector_;

			if (PseudoRandom.randDouble() < SwapProbability) {
				int pos1;
				int pos2;

				pos1 = PseudoRandom.randInt(0, permutationLength - 1);
				pos2 = PseudoRandom.randInt(0, permutationLength - 1);

				while (pos1 == pos2) {
					if (pos1 == (permutationLength - 1))
						pos2 = PseudoRandom.randInt(0, permutationLength - 2);
					else
						pos2 = PseudoRandom.randInt(pos1, permutationLength - 1);
				} // while
					// swap
				int temp = permutation[pos1];
				permutation[pos1] = permutation[pos2];
				permutation[pos2] = temp;
			} // if
		} // if
		else {
			Configuration.logger_.severe("SwapMutation.doMutation: invalid type. " + ""
					+ solution.getDecisionVariables()[0].getVariableType());

			Class cls = java.lang.String.class;
			String name = cls.getName();
			throw new JMException("Exception in " + name + ".doMutation()");
		}
		for (int i = 1; i < solution.getDecisionVariables().length; i++) {
			Binary binaryVariable = (Binary) solution.getDecisionVariables()[i];
			for (int j = 0; j < binaryVariable.getNumberOfBits(); j++)
				if (PseudoRandom.randDouble() < binaryProbability)
					binaryVariable.bits_.flip(j);
		}
		// BitFlip mutation applied to the binary part

	} // doMutation
} // PolynomialBitFlipMutation
