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

public class SwapMutation_mod extends Mutation {
	private static final double ETA_M_DEFAULT_ = 20.0;
	private final double eta_m_=ETA_M_DEFAULT_;
	
  private Double SwapMutationProbability_ = null ;
  private double distributionIndex_ = eta_m_;

  /**
   * Valid solution types to apply this operator 
   */
	private static final List VALID_TYPES = Arrays.asList(PermutationBinarySolutionType.class) ;

  /**
   * Constructor
   */
  public SwapMutation_mod(HashMap<String, Object> parameters) {
		super(parameters) ;
  	if (parameters.get("SwapMutationProbability") != null)
  		SwapMutationProbability_ = (Double) parameters.get("SwapMutationProbability") ;  				
  	if (parameters.get("distributionIndex") != null)
  		distributionIndex_ = (Double) parameters.get("distributionIndex") ;  		
	} // PolynomialBitFlipMutation
	
	@Override
  public Object execute(Object object) throws JMException {
		Solution solution = (Solution)object;

		if (!VALID_TYPES.contains(solution.getType().getClass())) {
			Configuration.logger_.severe("SwapMutation.execute: the solution " +
					"type " + solution.getType() + " is not allowed with this operator");

			Class cls = java.lang.String.class;
			String name = cls.getName(); 
			throw new JMException("Exception in " + name + ".execute()") ;
		} // if 

		doMutation(SwapMutationProbability_, solution);
		return solution;
  } // execute

	/**
	 * doMutation method
	 * @param realProbability
	 * @param binaryProbability
	 * @param solution
	 * @throws JMException
	 */
	public void doMutation(Double SwapProbability, Solution solution) throws JMException {   
		double rnd, delta1, delta2, mut_pow, deltaq;
		double y, yl, yu, val, xy;
		

		
		// Polynomial mutation applied to the array real
	    int permutation[] ;
	    int permutationLength ;
		    if (solution.getType().getClass() == PermutationBinarySolutionType.class) {

		      permutationLength = ((Permutation)solution.getDecisionVariables()[0]).getLength() ;
		      permutation = ((Permutation)solution.getDecisionVariables()[0]).vector_ ;

		      if (PseudoRandom.randDouble() < SwapProbability) {
		        int pos1 ;
		        int pos2 ;
		        int pos_route ;
		        int low = 0 ;
		        int up = 0;
		        pos_route = PseudoRandom.randInt(0,permutationLength-1) ;
	    		String s = solution.getDecisionVariables()[1].toString() ; 
	    	    char [] s0 = s.toCharArray();
	    	    
	    	    if (pos_route==0) {
	    	    	low = 0;
	    	    }
	    	    else {
					for (int i = pos_route-1; i >= 0; i-- ) {
						if ((s0[i]-48) == 1) {
							low = i+1;
							break;
						}
						low = 0;
					}
	    	    }
				for (int i = pos_route; i < permutationLength-1 ; i++ ) {
					if ((s0[i]-48) == 1) {
						up = i+1;
						break;
					}
					up = permutationLength;
				}
				
		        
		        pos1 = PseudoRandom.randInt(low,up-1) ;
		        pos2 = PseudoRandom.randInt(low,up-1) ;


		        // swap
		        int temp = permutation[pos1];
		        permutation[pos1] = permutation[pos2];
		        permutation[pos2] = temp; 
		        
				//System.out.println( Integer.toString(up));
		      } // if

		    } // if
		    else  {
		      Configuration.logger_.severe("SwapMutation.doMutation: invalid type. " +
		          ""+ solution.getDecisionVariables()[0].getVariableType());

		      Class cls = java.lang.String.class;
		      String name = cls.getName(); 
		      throw new JMException("Exception in " + name + ".doMutation()") ;
		    }

		// BitFlip mutation applied to the binary part

	} // doMutation
} // PolynomialBitFlipMutation

