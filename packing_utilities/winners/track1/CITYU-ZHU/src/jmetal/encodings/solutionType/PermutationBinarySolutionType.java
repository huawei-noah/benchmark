//  MOEAD_SDVRP 
//  For EMO2021 Huawei VRP competition
//
//  Author:         LIU Fei
//  E-mail:         fliu36-c@my.cityu.edu.hk
//  Create Date:    2021.2.1
//  Last modified   Date: 2021.2.2
//

package jmetal.encodings.solutionType;

import jmetal.core.Problem;
import jmetal.core.SolutionType;
import jmetal.core.Variable;
import jmetal.encodings.variable.Binary;
import jmetal.encodings.variable.Permutation;


/**
 * Class representing  a solution type including two variables: an integer 
 * and a real.
 */
public class PermutationBinarySolutionType extends SolutionType {
	private final int permutationVariables_ ;
	private final int binaryVariables_ ;

	/**
	 * Constructor
	 * @param problem  Problem to solve
	 * @param intVariables Number of integer variables
	 * @param realVariables Number of real variables
	 */
	public PermutationBinarySolutionType(Problem problem, int permutationVariables, int binaryVariables) {
		super(problem) ;
		permutationVariables_ = permutationVariables ;
		binaryVariables_ = binaryVariables ;
	} // Constructor

	/**
	 * Creates the variables of the solution
	 * @throws ClassNotFoundException
	 */
	public Variable[] createVariables() throws ClassNotFoundException {
		Variable [] variables = new Variable[problem_.getNumberOfVariables()];

		for (int var = 0; var < permutationVariables_; var++)
		  variables[var] = new Permutation(problem_.getLength(var)); 
		
		for (int var = permutationVariables_; var < (permutationVariables_ + binaryVariables_ ); var++)
				variables[var] = new Binary(problem_.getLength(var));

		return variables ;
	} // createVariables
} // IntRealSolutionType
