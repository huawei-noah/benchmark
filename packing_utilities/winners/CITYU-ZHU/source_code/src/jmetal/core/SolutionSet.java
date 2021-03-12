//  SolutionSet.Java
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

package jmetal.core;

import jmetal.util.Configuration;
import jmetal.problems.CVRP_mix_integer;
import jmetal.core.Problem;
import jmetal.encodings.variable.Permutation;

import java.io.*;
import java.util.*;

/** 
 * Class representing a SolutionSet (a set of solutions)
 */
public class SolutionSet implements Serializable {

  /**
   * Stores a list of <code>solution</code> objects.
   */
  protected final List<Solution> solutionsList_;

  /** 
   * Maximum size of the solution set 
   */
  private int capacity_ = 0; 

  /**
   * Constructor.
   * Creates an unbounded solution set.
   */
  public SolutionSet() {
    solutionsList_ = new ArrayList<Solution>();
  } // SolutionSet

  /** 
   * Creates a empty solutionSet with a maximum capacity.
   * @param maximumSize Maximum size.
   */
  public SolutionSet(int maximumSize){    
    solutionsList_ = new ArrayList<Solution>();
    capacity_      = maximumSize;
  } // SolutionSet

  /** 
   * Inserts a new solution into the SolutionSet. 
   * @param solution The <code>Solution</code> to store
   * @return True If the <code>Solution</code> has been inserted, false 
   * otherwise. 
   */
  public boolean add(Solution solution) {
    if (solutionsList_.size() == capacity_) {
      Configuration.logger_.severe("The population is full");
      Configuration.logger_.severe("Capacity is : "+capacity_);
      Configuration.logger_.severe("\t Size is: "+ this.size());
      return false;
    } // if

    solutionsList_.add(solution);
    return true;
  } // add

  public boolean add(int index, Solution solution) {
    solutionsList_.add(index, solution) ;
    return true ;
  }
  /*
  public void add(Solution solution) {
    if (solutionsList_.size() == capacity_)
      try {
        throw new JMException("SolutionSet.Add(): the population is full") ;
      } catch (JMException e) {
        e.printStackTrace();
      }
    else
      solutionsList_.add(solution);
  }
  */
  /**
   * Returns the ith solution in the set.
   * @param i Position of the solution to obtain.
   * @return The <code>Solution</code> at the position i.
   * @throws IndexOutOfBoundsException Exception
   */
  public Solution get(int i) {
    if (i >= solutionsList_.size()) {
      throw new IndexOutOfBoundsException("Index out of Bound "+i);
    }
    return solutionsList_.get(i);
  } // get

  /**
   * Returns the maximum capacity of the solution set
   * @return The maximum capacity of the solution set
   */
  public int getMaxSize(){
    return capacity_ ;
  } // getMaxSize

  /** 
   * Sorts a SolutionSet using a <code>Comparator</code>.
   * @param comparator <code>Comparator</code> used to sort.
   */
  public void sort(Comparator comparator){
    if (comparator == null) {
      Configuration.logger_.severe("No criterium for comparing exist");
      return ;
    } // if
    Collections.sort(solutionsList_,comparator);
  } // sort

  /** 
   * Returns the index of the best Solution using a <code>Comparator</code>.
   * If there are more than one occurrences, only the index of the first one is returned
   * @param comparator <code>Comparator</code> used to compare solutions.
   * @return The index of the best Solution attending to the comparator or 
   * <code>-1<code> if the SolutionSet is empty
   */
   int indexBest(Comparator comparator){
    if ((solutionsList_ == null) || (this.solutionsList_.isEmpty())) {
      return -1;
    }

    int index = 0; 
    Solution bestKnown = solutionsList_.get(0), candidateSolution;
    int flag;
    for (int i = 1; i < solutionsList_.size(); i++) {        
      candidateSolution = solutionsList_.get(i);
      flag = comparator.compare(bestKnown, candidateSolution);
      if (flag == +1) {
        index = i;
        bestKnown = candidateSolution; 
      }
    }

    return index;
  } // indexBest


  /** 
   * Returns the best Solution using a <code>Comparator</code>.
   * If there are more than one occurrences, only the first one is returned
   * @param comparator <code>Comparator</code> used to compare solutions.
   * @return The best Solution attending to the comparator or <code>null<code>
   * if the SolutionSet is empty
   */
  public Solution best(Comparator comparator){
    int indexBest = indexBest(comparator);
    if (indexBest < 0) {
      return null;
    } else {
      return solutionsList_.get(indexBest);
    }

  } // best  


  /** 
   * Returns the index of the worst Solution using a <code>Comparator</code>.
   * If there are more than one occurrences, only the index of the first one is returned
   * @param comparator <code>Comparator</code> used to compare solutions.
   * @return The index of the worst Solution attending to the comparator or 
   * <code>-1<code> if the SolutionSet is empty
   */
  public int indexWorst(Comparator comparator){
    if ((solutionsList_ == null) || (this.solutionsList_.isEmpty())) {
      return -1;
    }

    int index = 0;
    Solution worstKnown = solutionsList_.get(0), candidateSolution;
    int flag;
    for (int i = 1; i < solutionsList_.size(); i++) {        
      candidateSolution = solutionsList_.get(i);
      flag = comparator.compare(worstKnown, candidateSolution);
      if (flag == -1) {
        index = i;
        worstKnown = candidateSolution;
      }
    }

    return index;

  } // indexWorst

  /** 
   * Returns the worst Solution using a <code>Comparator</code>.
   * If there are more than one occurrences, only the first one is returned
   * @param comparator <code>Comparator</code> used to compare solutions.
   * @return The worst Solution attending to the comparator or <code>null<code>
   * if the SolutionSet is empty
   */
  public Solution worst(Comparator comparator){

    int index = indexWorst(comparator);
    if (index < 0) {
      return null;
    } else {
      return solutionsList_.get(index);
    }

  } // worst


  /** 
   * Returns the number of solutions in the SolutionSet.
   * @return The size of the SolutionSet.
   */  
  public int size(){
    return solutionsList_.size();
  } // size

  /** 
   * Writes the objective function values of the <code>Solution</code> 
   * objects into the set in a file.
   * @param path The output file name
   */
  public void printObjectivesToFile(String path){
    try {
      /* Open the file */
      FileOutputStream fos   = new FileOutputStream(path)     ;
      OutputStreamWriter osw = new OutputStreamWriter(fos)    ;
      BufferedWriter bw      = new BufferedWriter(osw)        ;

      for (Solution aSolutionsList_ : solutionsList_) {
        //if (this.vector[i].getFitness()<1.0) {
        bw.write(aSolutionsList_.toString());
        bw.newLine();
        //}
      }

      /* Close the file */
      bw.close();
    }catch (IOException e) {
      Configuration.logger_.severe("Error acceding to the file");
      e.printStackTrace();
    }
  } // printObjectivesToFile

  /**
   * Writes the decision encodings.variable values of the <code>Solution</code>
   * solutions objects into the set in a file.
   * @param path The output file name
   */
  public void printVariablesToFile(String path){
    try {
      FileOutputStream fos   = new FileOutputStream(path)     ;
      OutputStreamWriter osw = new OutputStreamWriter(fos)    ;
      BufferedWriter bw      = new BufferedWriter(osw)        ;            

      if (size()>0) {
        int numberOfVariables = solutionsList_.get(0).getDecisionVariables().length ;
        for (Solution aSolutionsList_ : solutionsList_) {
          for (int j = 0; j < numberOfVariables; j++)
            bw.write(aSolutionsList_.getDecisionVariables()[j].toString() + " ");
          bw.newLine();
        }
      }
      bw.close();
    }catch (IOException e) {
      Configuration.logger_.severe("Error acceding to the file");
      e.printStackTrace();
    }       
  } // printVariablesToFile
  
  public void printPathToFile(String path,int Split_minweight){

	    try {
	    	  /** The capacity that all vehicles in fruitybun-data.vrp have. */
	    	  int[] VEHICLE_CAPACITY = {50,60,60};
	    	  //public static final int VEHICLE_CAPACITY2 = 100;

	    	  /** The number of nodes in the fruitybun CVRP i.e. the depot and the customers */
	    	  int NUM_trucks = 10;
	    	  int Type_trucks = 3;
	    	  int NUM_nodes = 10;
	    	  //int Split_minweight =10;
	    	  int numberOfCities_ = 0;
	    	  double Split_precision = 6.0;
	    	  int new_num_nodes = 0;


	    	 for (int i = 0; i < NUM_nodes; i++) {
	    	    	if  (node_weights[i] > Split_minweight) {
	    	        	new_num_nodes += 2; 

	    	    	}
	    	    	else {
	    	        	new_num_nodes += 1; 	
	    	    	} 	
	    	    }
	    	 
	    	numberOfCities_ = new_num_nodes;
	    	     
	        double[][] new_node_coords = new double[numberOfCities_][2];
	        double[] new_node_weights = new double[numberOfCities_];

 
            int no_results = 0;
            int x = 0 ;
	         int numberOfVariables = solutionsList_.get(0).getDecisionVariables().length ;
	          for (Solution aSolutionsList_ : solutionsList_) {
	        	
	  	      FileOutputStream fos   = new FileOutputStream(path+Integer.toString(no_results)+".dat")     ;
		      OutputStreamWriter osw = new OutputStreamWriter(fos)    ;
		      BufferedWriter bw      = new BufferedWriter(osw)        ;
		      no_results += 1;
		        int var = 2;
		        new_num_nodes = 0;
		        for (int i = 0; i < NUM_nodes; i++) {
		        	if  (node_weights[i] > Split_minweight) {
		        		String s = aSolutionsList_.getDecisionVariables()[var].toString() ; 
		        	    char [] s0 = s.toCharArray();
		        	    double s00 = 0;
		        	    for (int j=0;j<2;j++) {
		        	    	s00 += (s0[j]- 48)*2^j;
		        	    }
		        	    new_node_coords[new_num_nodes] = node_coords[i];
		            	new_node_weights[new_num_nodes] = node_weights[i]*(s00/Split_precision);
		            	new_num_nodes += 1; 
		        	    new_node_coords[new_num_nodes] = node_coords[i];
		            	new_node_weights[new_num_nodes] = node_weights[i]*(1.0-s00/Split_precision);
		            	new_num_nodes += 1; 
		        		
		        		var += 1;
		        	}
		        	else {
		        		new_node_coords[new_num_nodes] = node_coords[i];
		        		new_node_weights[new_num_nodes] = node_weights[i];    	
		        		new_num_nodes += 1; 	
		        	}
		        }
			  String if_start = aSolutionsList_.getDecisionVariables()[1].toString();
			  char [] if_start0 = if_start.toCharArray();
              bw.write( "bariables = x1, x2 ");
	          bw.newLine();
              bw.write( "zone ");
	          bw.newLine();
              bw.write( start_coords[0] + " "); 
              bw.write( start_coords[1] + " "); 
	          bw.newLine();
	          for (int j = 0; j < numberOfCities_; j++) {
	        	  
	        	  x = ((Permutation)aSolutionsList_.getDecisionVariables()[0]).vector_[j];
	        	  x = ((Permutation)aSolutionsList_.getDecisionVariables()[0]).vector_[j];
	        	  if (j>0 && (if_start0[j-1]-48)==1) {
		              bw.write( end_coords[0] + " "); 
		              bw.write( end_coords[1] + " ");
			          bw.newLine();
		              bw.write( "zone ");
			          bw.newLine();
		              bw.write( start_coords[0] + " "); 
		              bw.write( start_coords[1] + " "); 
			          bw.newLine();
	        	  }
	              bw.write( new_node_coords[x][0] + " "); 
	              bw.write( new_node_coords[x][1] + " "); 
		          bw.newLine();
	          }
              bw.write( end_coords[0] + " "); 
              bw.write( end_coords[1] + " ");

		      bw.close();
	        }


	    }catch (IOException e) {
	      Configuration.logger_.severe("Error acceding to the file");
	      e.printStackTrace();
	    } 

	  } // printVariablesToFile 
  
  public HashMap getresults(){
	     HashMap output ;
	     output = new HashMap();
	     output.put("result",solutionsList_);
	     return output;
	  } // printVariablesToFile  
  
  public void printFinalPathToFile(String path,double[] all_split_minv, ArrayList<Solution> solutionsList_0,int[] no_PF,int[] VEHICLE_CAPACITY,int NUM_nodes){

	    try {
	    	  /** The capacity that all vehicles in fruitybun-data.vrp have. */
	    	  //int[] VEHICLE_CAPACITY = {50,60,60};
	    	  //public static final int VEHICLE_CAPACITY2 = 100;

	    	  /** The number of nodes in the fruitybun CVRP i.e. the depot and the customers */
	    	  int NUM_trucks = 10;
	    	  int Type_trucks = VEHICLE_CAPACITY.length;
	    	  //int NUM_nodes = 10;
	    	  //int Split_minweight =10;
	    	  //int numberOfCities_ = 0;
	    	  double Split_precision = 6.0;

	    	
	    	  int no_results = 0;
	    	  int x = 0 ;
	          int numberOfVariables = solutionsList_0.get(0).getDecisionVariables().length ;
	          for (Solution aSolutionsList_ : solutionsList_0) {
	        	  
		    	  int new_num_nodes = 0;
		    	  int numberOfCities_ = 0;


		    	 for (int i = 0; i < NUM_nodes; i++) {
		    	    	if  (node_weights[i] > all_split_minv[no_PF[no_results]]) {
		    	        	new_num_nodes += 2; 

		    	    	}
		    	    	else {
		    	        	new_num_nodes += 1; 	
		    	    	} 	
		    	    }
		    	 
		    	numberOfCities_ = new_num_nodes;
		    	     
		        double[][] new_node_coords = new double[numberOfCities_][2];
		        double[] new_node_weights = new double[numberOfCities_];
        	
		        FileOutputStream fos   = new FileOutputStream(path+Integer.toString(no_results)+"_w.dat")     ;
		        OutputStreamWriter osw = new OutputStreamWriter(fos)    ;
		        BufferedWriter bw      = new BufferedWriter(osw)        ;
		        FileOutputStream fos0   = new FileOutputStream(path+Integer.toString(no_results)+".dat")     ;
		        OutputStreamWriter osw0 = new OutputStreamWriter(fos0)    ;
		        BufferedWriter bw0      = new BufferedWriter(osw0)        ;
		        
		        int var = 2;
		        new_num_nodes = 0;
		        for (int i = 0; i < NUM_nodes; i++) {
		        	if  (node_weights[i] >  all_split_minv[no_PF[no_results]]) {
		        		String s = aSolutionsList_.getDecisionVariables()[var].toString() ; 
		        	    char [] s0 = s.toCharArray();
		        	    double s00 = 0;
		        	    for (int j=0;j<2;j++) {
		        	    	s00 += (s0[j]- 48)*2^j;
		        	    }
		        	    new_node_coords[new_num_nodes] = node_coords[i];
		            	new_node_weights[new_num_nodes] = node_weights[i]*(s00/Split_precision);
		            	new_num_nodes += 1; 
		        	    new_node_coords[new_num_nodes] = node_coords[i];
		            	new_node_weights[new_num_nodes] = node_weights[i]*(1.0-s00/Split_precision);
		            	new_num_nodes += 1; 
		        		
		        		var += 1;
		        	}
		        	else {
		        		new_node_coords[new_num_nodes] = node_coords[i];
		        		new_node_weights[new_num_nodes] = node_weights[i];    	
		        		new_num_nodes += 1; 	
		        	}
		        }
		        
		        no_results += 1;
		        
			  String if_start = aSolutionsList_.getDecisionVariables()[1].toString();
			  char [] if_start0 = if_start.toCharArray();
			  bw.write( "bariables = x1, x2, w ");
	          bw.newLine();
	          bw.write( "zone ");
	          bw.newLine();
	          bw.write( start_coords[0] + " "); 
	          bw.write( start_coords[1] + " ");
	          bw.write( 0 + " ");
	          bw.newLine();
			  bw0.write( "bariables = x1, x2 ");
	          bw0.newLine();
	          bw0.write( "zone ");
	          bw0.newLine();
	          bw0.write( start_coords[0] + " "); 
	          bw0.write( start_coords[1] + " ");
	          bw0.newLine();
	          int n_route = 0;
	          double[] w_route = new double[100];
	          for (int j = 0; j < numberOfCities_; j++) {
	        	  
	        	  //x = ((Permutation)aSolutionsList_.getDecisionVariables()[0]).vector_[j];
	        	  x = ((Permutation)aSolutionsList_.getDecisionVariables()[0]).vector_[j];
	        	  if (j>0 && (if_start0[j-1]-48)==1) {
		              bw.write( end_coords[0] + " "); 
		              bw.write( end_coords[1] + " ");
		              bw.write( 0 + " ");
			          bw.newLine();
		              bw.write( "zone ");
			          bw.newLine();
		              bw.write( start_coords[0] + " "); 
		              bw.write( start_coords[1] + " ");
		              bw.write( 0 + " ");
			          bw.newLine();
		              bw0.write( end_coords[0] + " "); 
		              bw0.write( end_coords[1] + " ");
			          bw0.newLine();
		              bw0.write( "zone ");
			          bw0.newLine();
		              bw0.write( start_coords[0] + " "); 
		              bw0.write( start_coords[1] + " ");
			          bw0.newLine();
			          n_route += 1;
	        	  }
	              bw.write( new_node_coords[x][0] + " "); 
	              bw.write( new_node_coords[x][1] + " "); 
	              bw.write( new_node_weights[x] + " "); 	           
		          bw.newLine();
	              bw0.write( new_node_coords[x][0] + " "); 
	              bw0.write( new_node_coords[x][1] + " "); 
	              //bw.write( new_node_weights[j] + " "); 	           
		          bw0.newLine();
		          w_route[n_route] += new_node_weights[j];
	          }
	          bw.write( end_coords[0] + " "); 
	          bw.write( end_coords[1] + " ");
	          bw.write( 0 + " ");
		      bw.close();
	          bw0.write( end_coords[0] + " "); 
	          bw0.write( end_coords[1] + " ");
		      bw0.close();
	        }

          FileOutputStream fos   = new FileOutputStream("Finalvar.dat")     ;
          OutputStreamWriter osw = new OutputStreamWriter(fos)    ;
          BufferedWriter bw      = new BufferedWriter(osw)        ;            

          if (size()>0) {
            for (Solution aSolutionsList_ : solutionsList_0) {
              int numberOfVariables1 = aSolutionsList_.getDecisionVariables().length ;
              for (int j = 0; j < numberOfVariables1; j++)
                bw.write(aSolutionsList_.getDecisionVariables()[j].toString() + " ");
              bw.newLine();
            }
          }
          bw.close();
          
          fos   = new FileOutputStream("Finalobj.dat")     ;  
          osw   =  new OutputStreamWriter(fos)    ;
          bw    = new BufferedWriter(osw)        ; 

          if (size()>0) {
              for (Solution aSolutionsList_ : solutionsList_0) {
                  //if (this.vector[i].getFitness()<1.0) {
                  bw.write(aSolutionsList_.toString());
                  bw.newLine();
                  //}
                }
          }
          bw.close();
          

	          
	    }catch (IOException e) {
	      Configuration.logger_.severe("Error acceding to the file");
	      e.printStackTrace();
	    } 


	  } // printVariablesToFile  


  /**
   * Write the function values of feasible solutions into a file
   * @param path File name
   */
  public void printFeasibleFUN(String path) {
    try {
      FileOutputStream fos   = new FileOutputStream(path)     ;
      OutputStreamWriter osw = new OutputStreamWriter(fos)    ;
      BufferedWriter bw      = new BufferedWriter(osw)        ;

      for (Solution aSolutionsList_ : solutionsList_) {
        if (aSolutionsList_.getOverallConstraintViolation() == 0.0) {
          bw.write(aSolutionsList_.toString());
          bw.newLine();
        }
      }
      bw.close();
    }catch (IOException e) {
      Configuration.logger_.severe("Error acceding to the file");
      e.printStackTrace();
    }
  }

  /**
   * Write the encodings.variable values of feasible solutions into a file
   * @param path File name
   */
  public void printFeasibleVAR(String path) {
    try {
      FileOutputStream fos   = new FileOutputStream(path)     ;
      OutputStreamWriter osw = new OutputStreamWriter(fos)    ;
      BufferedWriter bw      = new BufferedWriter(osw)        ;            

      if (size()>0) {
        int numberOfVariables = solutionsList_.get(0).getDecisionVariables().length ;
        for (Solution aSolutionsList_ : solutionsList_) {
          if (aSolutionsList_.getOverallConstraintViolation() == 0.0) {
            for (int j = 0; j < numberOfVariables; j++)
              bw.write(aSolutionsList_.getDecisionVariables()[j].toString() + " ");
            bw.newLine();
          }
        }
      }
      bw.close();
    }catch (IOException e) {
      Configuration.logger_.severe("Error acceding to the file");
      e.printStackTrace();
    } 
  }
  
  /** 
   * Empties the SolutionSet
   */
  public void clear(){
    solutionsList_.clear();
  } // clear

  /** 
   * Deletes the <code>Solution</code> at position i in the set.
   * @param i The position of the solution to remove.
   */
  public void remove(int i){        
    if (i > solutionsList_.size()-1) {            
      Configuration.logger_.severe("Size is: "+this.size());
    } // if
    solutionsList_.remove(i);    
  } // remove


  /**
   * Returns an <code>Iterator</code> to access to the solution set list.
   * @return the <code>Iterator</code>.
   */    
  public Iterator<Solution> iterator(){
    return solutionsList_.iterator();
  } // iterator   

  /** 
   * Returns a new <code>SolutionSet</code> which is the result of the union
   * between the current solution set and the one passed as a parameter.
   * @param solutionSet SolutionSet to join with the current solutionSet.
   * @return The result of the union operation.
   */
  public SolutionSet union(SolutionSet solutionSet) {       
    //Check the correct size. In development
    int newSize = this.size() + solutionSet.size();
    if (newSize < capacity_)
      newSize = capacity_;

    //Create a new population 
    SolutionSet union = new SolutionSet(newSize);                
    for (int i = 0; i < this.size(); i++) {      
      union.add(this.get(i));
    } // for

    for (int i = this.size(); i < (this.size() + solutionSet.size()); i++) {
      union.add(solutionSet.get(i-this.size()));
    } // for

    return union;        
  } // union                   

  /** 
   * Replaces a solution by a new one
   * @param position The position of the solution to replace
   * @param solution The new solution
   */
  public void replace(int position, Solution solution) {
    if (position > this.solutionsList_.size()) {
      solutionsList_.add(solution);
    } // if 
    solutionsList_.remove(position);
    solutionsList_.add(position,solution);
  } // replace

  /**
   * Copies the objectives of the solution set to a matrix
   * @return A matrix containing the objectives
   */
  public double [][] writeObjectivesToMatrix() {
    if (this.size() == 0) {
      return null;
    }
    double [][] objectives;
    objectives = new double[size()][get(0).getNumberOfObjectives()];
    for (int i = 0; i < size(); i++) {
      for (int j = 0; j < get(0).getNumberOfObjectives(); j++) {
        objectives[i][j] = get(i).getObjective(j);
      }
    }
    return objectives;
  } // writeObjectivesMatrix

  public void printObjectives() {
    for (int i = 0; i < solutionsList_.size(); i++)
      System.out.println(""+ solutionsList_.get(i)) ;
  }

  public void setCapacity(int capacity) {
    capacity_ = capacity ;
  }

  public int getCapacity() {
    return capacity_ ;
  }
  
  private static double [] start_coords = new double[] {40,40};
  private static double [] end_coords = new double[] {40,40};
  private static double [][] node_coords = new double[][] { // dummy entry to make index of array match indexing of nodes in fruitybun-data.vrp
  	  {22, 22}, // the coordinates of node 2 ...
  	  {36, 26},
  	  {21, 45},
  	  {45, 35},
  	  {55, 20},
  	  {33, 34},
  	  {50, 50},
  	  {55, 45},
  	  {26, 59}, // node 10
  	  {40, 66},
  	  {55, 65},
  	  {35, 51},
  	  {62, 35},
  	  {62, 57},
  	  {62, 24},
  	  {21, 36},
  	  {33, 44},
  	  {9, 56},
  	  {62, 48}, // node 20
  	  {66, 14},
  	  {44, 13},
  	  {26, 13},
  	  {11, 28},
  	  {7, 43},
  	  {17, 64},
  	  {41, 46},
  	  {55, 34},
  	  {35, 16},
  	  {52, 26}, // node 30
  	  {43, 26},
  	  {31, 76},
  	  {22, 53},
  	  {26, 29},
  	  {50, 40},
  	  {55, 50},
  	  {54, 10},
  	  {60, 15},
  	  {47, 66},
  	  {30, 60}, // node 40
  	  {30, 50},
  	  {12, 17},
  	  {15, 14},
  	  {16, 19},
  	  {21, 48},
  	  {50, 30},
  	  {51, 42},
  	  {50, 15},
  	  {48, 21},
  	  {12, 38}, // node 50
  	  {15, 56},
  	  {29, 39},
  	  {54, 38},
  	  {55, 57},
  	  {67, 41},
  	  {10, 70},
  	  {6, 25},
  	  {65, 27},
  	  {40, 60},
  	  {70, 64}, // node 60
  	  {64, 4},
  	  {36, 6},
  	  {30, 20},
  	  {20, 30},
  	  {15, 5},
  	  {50, 70},
  	  {57, 72},
  	  {45, 42},
  	  {38, 33},
  	  {50, 4},  // node 70
  	  {66, 8},
  	  {59, 5},
  	  {35, 60},
  	  {27, 24},
  	  {40, 20},
  	  {40, 37}};// node 76	  
  		  
  		  
  private static double[] node_weights = new double[]
  			{  // dummy entry to make index of array match indexing of nodes in fruitybun-data.vrp
  					 18, // node 2
  					 26, // node 3
  					 11, // node 4
  					 30, // node 5
  					 21, // node 6
  					 19, // node 7
  					 15, // node 8
  					 16, // node 9
  					 29, // node 10
  					 26, // node 11
  					 37, // node 12
  					 16, // node 13
  					 12, // node 14
  					 31, // node 15
  					 8,  // node 16
  					 19, // node 17
  					 20, // node 18
  					 13, // node 19
  					 15, // node 20
  					 22, // node 21
  					 28, // node 22
  					 12, // node 23
  					 6,  // node 24
  					 27, // node 25
  					 14, // node 26
  					 18, // node 27
  					 17, // node 28
  					 29, // node 29
  					 13, // node 30
  					 22, // node 31
  					 25, // node 32
  					 28, // node 33
  					 27, // node 34
  					 19, // node 35
  					 10, // node 36
  					 12, // node 37
  					 14, // node 38
  					 24, // node 39
  					 16, // node 40
  					 33, // node 41
  					 15, // node 42
  					 11, // node 43
  					 18, // node 44
  					 17, // node 45
  					 21, // node 46
  					 27, // node 47
  					 19, // node 48
  					 20, // node 49
  					 5,  // node 50
  					 22, // node 51
  					 12, // node 52
  					 19, // node 53
  					 22, // node 54
  					 16, // node 55
  					 7,  // node 56
  					 26, // node 57
  					 14, // node 58
  					 21, // node 59
  					 24, // node 60
  					 13, // node 61
  					 15, // node 62
  					 18, // node 63
  					 11, // node 64
  					 28, // node 65
  					 9,  // node 66
  					 37, // node 67
  					 30, // node 68
  					 10, // node 69
  					 8,  // node 70
  					 11, // node 71
  					 3,  // node 72
  					 1,  // node 73
  					 6,  // node 74
  					 10, // node 75
  					 20};// node 76  
 // mTSP



} // SolutionSet


