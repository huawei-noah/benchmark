//  IntRealSolutionType.java
//
//  Author:
//       Antonio J. Nebro <antonio@lcc.uma.es>
//       Juan J. Durillo <durillo@lcc.uma.es>
// 
//  Copyright (c) 2011 Antonio J. Nebro, Juan J. Durillo
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU Lesser General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
// 
//  You should have received a copy of the GNU Lesser General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

package jmetal.encodings.solutionType;

import jmetal.core.Problem;
import jmetal.core.SolutionType;
import jmetal.core.Variable;
import jmetal.encodings.variable.Binary;
import jmetal.encodings.variable.Int;
import jmetal.encodings.variable.Permutation;


/**
 * Class representing  a solution type including two variables: an integer 
 * and a real.
 */
public class PermutationIntSolutionType extends SolutionType {
	private final int permutationVariables_ ;
	private final int intVariables_ ;

	/**
	 * Constructor
	 * @param problem  Problem to solve
	 * @param intVariables Number of integer variables
	 * @param realVariables Number of real variables
	 */
	public PermutationIntSolutionType(Problem problem, int permutationVariables, int intVariables) {
		super(problem) ;
		permutationVariables_ = permutationVariables ;
		intVariables_ = intVariables ;
	} // Constructor

	/**
	 * Creates the variables of the solution
	 * @throws ClassNotFoundException
	 */
	public Variable[] createVariables() throws ClassNotFoundException {
		Variable [] variables = new Variable[problem_.getNumberOfVariables()];

		for (int var = 0; var < permutationVariables_; var++)
		  variables[var] = new Permutation(problem_.getLength(var)); 
		
		for (int var = permutationVariables_; var < (permutationVariables_ + intVariables_ ); var++)

			variables[var] = new Int((int)problem_.getLowerLimit(var),
					(int)problem_.getUpperLimit(var));    

		return variables ;
	} // createVariables
} // IntRealSolutionType
