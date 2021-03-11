//  MOEAD_SDVRP 
//  For EMO2021 Huawei VRP competition
//
//  Author:         LIU Fei  
//  E-mail:         fliu36-c@my.cityu.edu.hk
//  Create Date:    2021.2.1
//  Last modified   Date: 2021.2.15
//

package jmetal.problems;

import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.encodings.solutionType.PermutationSolutionType;
import jmetal.encodings.solutionType.PermutationBinarySolutionType;
import jmetal.encodings.variable.Permutation;
import jmetal.util.JMException;
import jmetal.encodings.variable.Binary;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;

import com.my.vrp.Node;

/**
 * Class representing a multi-objective TSP (Traveling Salesman Problem) problem.
 * This class is tested with two objectives and the KROA150 and KROB150 
 * instances of TSPLIB
 */
public class CVRP_mix_integer extends Problem {

  public int         numberOfCities_ ; 
  public double [][] distanceMatrix_ ;
  public double [][] costMatrix_;
  public int [] new_node_no;
  public double [] new_node_v;
  
  /** The capacity that all vehicles in fruitybun-data.vrp have. */
  public static double[] VEHICLE_CAPACITY ;
  public static double[] VEHICLE_VOLUME ;
  public static double[][] nodes_v_w;
  public static HashMap<String, Double> Distances;
  //public static final int VEHICLE_CAPACITY2 = 100;

  /** The number of nodes in the fruitybun CVRP i.e. the depot and the customers */
  public static final int NUM_trucks = 10;
  //public static final int Type_trucks = 3;
  public static int NUM_nodes;
  public static double Split_minv = 0;
  public static int If_check = 0;
  public static final double Split_precision = 6.0;
  public static double Relax_ratio = 0;
  public static double Relax_volume = 0;
  public static double[] truck_weight_ratio;
  ArrayList<Node> clients;
  public int[] if_large_box;
  public double[] if_hard_node;
  
  
  
  
 /** VEHICLE_VOLUME,NUM_nodes,Distances,client_v_w
  * Creates a new mTSP problem instance. It accepts data files from TSPLIB
  */
  public CVRP_mix_integer(String solutionType,
              double split_minv,
              double[] VEHICLE_CAPACITY0,
              double[] VEHICLE_VOLUME0,
              int NUM_nodes0,
              HashMap<String, Double> Distances0,
              double[][] nodes_v_w0,
              double ralax_ratio0,
              double[] truck_weight_ratio0,
              ArrayList<Node> clients0,
              int[] if_large_box0,
              double[] if_hard_node0
              ) throws IOException {
    //numberOfVariables_  = 2;
	Split_minv = split_minv;
    numberOfObjectives_ = 2;
    numberOfConstraints_= 2;
    problemName_        = "CVRP_mix";
    VEHICLE_CAPACITY = VEHICLE_CAPACITY0;
    VEHICLE_VOLUME = VEHICLE_VOLUME0; 
    nodes_v_w = nodes_v_w0;
    NUM_nodes = NUM_nodes0;
    Distances = Distances0;
    Relax_ratio = ralax_ratio0;
    Relax_volume = Relax_ratio*VEHICLE_VOLUME[0];
    truck_weight_ratio = truck_weight_ratio0;
    //If_check = if_check;
    clients = clients0;
    if_large_box = if_large_box0;
    if_hard_node = if_hard_node0;


    int new_num_nodes = 0;
             
//    length_       = new int[numberOfVariables_];
    
    for (int i = 0; i < NUM_nodes; i++) {
    	while (nodes_v_w[i][0]> 1.1*VEHICLE_VOLUME[VEHICLE_CAPACITY.length-1]) {
    		nodes_v_w[i][0] -= truck_weight_ratio[i]*VEHICLE_VOLUME[VEHICLE_CAPACITY.length-1];	
    	}
    	if  (nodes_v_w[i][0] > Split_minv) {//要不要分割。
        	new_num_nodes += 2; 
    	}
    	else {
        	new_num_nodes += 1; 	
    	} 
    	//System.out.println(nodes_v_w[i][0]);
    }

     numberOfVariables_ = new_num_nodes - NUM_nodes + 2;//至少一個節點。
     length_       = new int[numberOfVariables_];
     length_      [0] = new_num_nodes;
     length_      [1] = new_num_nodes-1;
     for (int i = 2; i < numberOfVariables_; i++) {
    	 length_      [i] = 2; 
     }
     

     numberOfCities_ = new_num_nodes;
     
     if (solutionType.compareTo("PermutationBinary") == 0)
     	solutionType_ = new PermutationBinarySolutionType(this,1,numberOfVariables_-1) ;
     else {
     	System.out.println("Error: solution type " + solutionType + " invalid") ;
     	System.exit(-1) ;
     }
  } // mTSP    
 /** 
  * Evaluates a solution 
  * @param solution The solution to evaluate
  */      
  public void evaluate(Solution solution) {
    double fitness1   ;
    double fitness2   ;
    double weights_precent ;
    int n_truck_used ;
//    double w_const ;
    int new_num_nodes = 0;
   
    
    fitness1   = 0.0 ;
    fitness2   = 0.0 ;
    weights_precent = 0.0;//當前的載重。
    n_truck_used = 0;
//    w_const = 0;
    int Type_trucks = VEHICLE_CAPACITY.length;
    
       
    //new_node_coords = new double[numberOfCities_][2];
    new_node_no = new int [numberOfCities_];//新節點的編號。。
    new_node_v = new double[numberOfCities_];//新節點的體積。

    int var = 2;
    for (int i = 0; i < NUM_nodes; i++) {
    	if  (nodes_v_w[i][0] > Split_minv) {//有新增的節點
    		String s = solution.getDecisionVariables()[var].toString() ; 
    	    char [] s0 = s.toCharArray();
    	    double s00 = 0;
    	    for (int j=0;j<2;j++) {
    	    	s00 += (s0[j]- 48)*Math.pow(2, j);//將二進制轉換為整數。0,1,2,3
    	    	//System.out.println((s0[j]- 48)*2^j);
    	    }
    	    //new_node_coords[new_num_nodes] = node_coords[i];
    	    new_node_no[new_num_nodes] = i+1;//新增一個解點，
        	new_node_v[new_num_nodes] = nodes_v_w[i][0]*(s00/Split_precision);//新增節點對應的體積。
        	new_num_nodes += 1; 
    	    //new_node_coords[new_num_nodes] = node_coords[i];
        	new_node_no[new_num_nodes] = i+1;
        	new_node_v[new_num_nodes] = nodes_v_w[i][0]*(1.0-s00/Split_precision);//6.0
        	new_num_nodes += 1; 
    		
    		var += 1;
    	}
    	else {//如果這個節點的體積很小，一個車子可以裝下。沒有新增的節點。
    		//new_node_coords[new_num_nodes] = node_coords[i];
    		new_node_no[new_num_nodes] = i+1;//這個節點的編號。
    		new_node_v[new_num_nodes] = nodes_v_w[i][0];    //這個節點的體積。	
    		new_num_nodes += 1; //下一個節點。	
    	}

    }    
     
    
	String if_start = solution.getDecisionVariables()[1].toString();
	char [] if_start0 = if_start.toCharArray();
	
	int[] route_use_large = new int[numberOfCities_];
	double[] route_hard = new double[numberOfCities_];
	int nroute = 0;
    for (int i = 0; i < (numberOfCities_ ); i++) {
    	route_hard[i] = 1.0;

        int x ; 
        x = ((Permutation)solution.getDecisionVariables()[0]).vector_[i] ;  
        if(i>0) {
            if (  (if_start0[i-1]- 48) == 1) {
            	nroute += 1;
            }
        }
        if ( if_large_box[new_node_no[x]-1]==1) {
        	route_use_large[nroute] = 1;
        }
        if (if_hard_node[new_node_no[x]-1]<route_hard[nroute]) {
        	route_hard[nroute] = if_hard_node[new_node_no[x]-1];
        }
        //route_hard[nroute] =  route_hard[nroute]*if_hard_node[new_node_no[x]-1];
    }
	
	
    double VEHICLE_CAPACITY_precent;
    VEHICLE_CAPACITY_precent = 0;
    
    nroute = 0;
    for (int i = 0; i < (numberOfCities_ ); i++) {
      int x ; 
      int x_last;

      
      x = ((Permutation)solution.getDecisionVariables()[0]).vector_[i] ;   
      if (i==0) {
    	  fitness1 += distance2_start_end(0,x);//從起點到x這個節點的距離。
    	  weights_precent = new_node_v[x];//這個節點的體積。
      }
      else {
          x_last = ((Permutation)solution.getDecisionVariables()[0]).vector_[i-1] ; //上一個節點。
          //System.out.println(if_start0[i-1]);
          if ( (if_start0[i-1]- 48) == 1) {//是否開始節點。
        	  fitness1 += distance2_start_end(NUM_nodes+1,x_last);
        	  fitness1 += distance2_start_end(0,x);
        	  if (route_use_large[nroute] == 1) {
        		  VEHICLE_CAPACITY_precent = (VEHICLE_VOLUME[Type_trucks-1]*Relax_ratio);
        	  }
        	  else {
            	  for (int j = 0; j < Type_trucks; j++) {
            		  if(route_hard[nroute]*(VEHICLE_VOLUME[j]*Relax_ratio)>weights_precent) {
            			  VEHICLE_CAPACITY_precent = route_hard[nroute]*(VEHICLE_VOLUME[j]*Relax_ratio);
            			  break;
            		  }
            		  VEHICLE_CAPACITY_precent = route_hard[nroute]*(VEHICLE_VOLUME[j]*Relax_ratio);
            	     // System.out.println(VEHICLE_CAPACITY_precent );
            	      //System.out.println(VEHICLE_CAPACITY[j]);
            	  }
        	  }

        	  fitness2 += weights_precent/VEHICLE_CAPACITY_precent;
        	  n_truck_used += 1;
//        	  if (weights_precent > VEHICLE_CAPACITY_precent) {
//        		  w_const += 1;
//        	  }
        	  weights_precent = new_node_v[x];
        	  nroute +=1;
          }
          else {
        	  fitness1 += distance2(x,x_last);
        	  weights_precent += new_node_v[x];
          }
      }
      //System.out.println(weights_precent);
      if (i==(numberOfCities_-1 )) {
    	  if (route_use_large[nroute] == 1) {
    		  VEHICLE_CAPACITY_precent = (VEHICLE_VOLUME[Type_trucks-1]*Relax_ratio);
    	  }
    	  else {
	    	  for (int j = 0; j < Type_trucks; j++) {
	    		  if(route_hard[nroute]*(VEHICLE_VOLUME[j]*Relax_ratio)>weights_precent) {
	    			  VEHICLE_CAPACITY_precent = route_hard[nroute]*(VEHICLE_VOLUME[j]*Relax_ratio);
	    			  break;
	    		  }
	    		  VEHICLE_CAPACITY_precent = route_hard[nroute]*(VEHICLE_VOLUME[j]*Relax_ratio);
	    	      //System.out.println(VEHICLE_CAPACITY_precent );
	    	     // System.out.println(VEHICLE_CAPACITY[j]);
	    	  }
    	  }
    	  fitness1 += distance2_start_end(NUM_nodes+1,x);
    	  fitness2 += weights_precent/VEHICLE_CAPACITY_precent;
    	  n_truck_used += 1;
//    	  if (weights_precent > VEHICLE_CAPACITY_precent) {
//    		  w_const += 1;
//    	  }
      }
      //System.out.println(VEHICLE_CAPACITY.toString());
//  cout << "I : " << i << ", x = " << x << ", y = " << y << endl ; 
    //System.out.println(weights_precent);
    } // for
    //fitness1 = fitness1;
    //fitness2 = 1-fitness2/n_truck_used;
    fitness1 = fitness1/100000;
    fitness2 = 1-fitness2/n_truck_used;

    /*
    if(fitness2 < 0) {
    	System.out.println(n_truck_used);
		for(int i=0;i<VEHICLE_VOLUME.length;i++) {
	    	System.out.println(VEHICLE_VOLUME[i]);
		  }
    	System.out.println(VEHICLE_CAPACITY_precent);
    	System.out.println(w_const);
    	System.out.println(Ralax_ratio);
    	System.out.println("check");
    }
    */
    //fitness2 = n_truck_used;
    
    solution.setObjective(0, fitness1);            
    solution.setObjective(1, fitness2);
  } // evaluate
  
  
  
  public void evaluateConstraints(Solution solution) throws JMException {

	    double weights_precent ;
//	    int n_truck_used ;
	    double w_const ;
	    double first_const ;
	    double w_all = 0.0;
//	    int new_num_nodes = 0;

	    weights_precent = 0.0;
//	    n_truck_used = 0;
	    w_const = 0;
	    first_const = 0;
	    int Type_trucks = VEHICLE_CAPACITY.length;
	    
		String if_start = solution.getDecisionVariables()[1].toString();
		char [] if_start0 = if_start.toCharArray();
		
	    
		int[] route_use_large = new int[numberOfCities_];
		double[] route_hard = new double[numberOfCities_];
		int nroute = 0;
	    for (int i = 0; i < (numberOfCities_ ); i++) {
	    	route_hard[i] = 1.0;

	        int x ; 
	        x = ((Permutation)solution.getDecisionVariables()[0]).vector_[i] ;  
	        if(i>0) {
	            if (  (if_start0[i-1]- 48) == 1) {
	            	nroute += 1;
	            }
	        }
	        if ( if_large_box[new_node_no[x]-1]==1) {
	        	route_use_large[nroute] = 1;
	        }
	        if (if_hard_node[new_node_no[x]-1]<route_hard[nroute]) {
	        	route_hard[nroute] = if_hard_node[new_node_no[x]-1];
	        }
	        //route_hard[nroute] =  route_hard[nroute]*if_hard_node[new_node_no[x]-1];
	    }
		
	    nroute = 0;
	    
	    for (int i = 1; i < numberOfCities_; i++) {
	    	//System.out.println(clients.get(new_node_no[i]-1).isMustFirst());
	    	int x;
	    	int x_last;
	    	x = ((Permutation)solution.getDecisionVariables()[0]).vector_[i] ;
	    	x_last = ((Permutation)solution.getDecisionVariables()[0]).vector_[i-1] ; 
	    	if (clients.get(new_node_no[x]-1).isMustFirst()) {
	    		int check_last;
	    		if(i-2>=0) {
	    			check_last = (if_start0[i-2]- 48);
	    		}
	    		else {
	    			check_last = 1;
	    		}
	    		if((if_start0[i-1]- 48) == 1 || (check_last == 1 && new_node_no[x]==new_node_no[x_last])){
	    			first_const += 0;
	    		}
	    		else {
	    			first_const += 100000000000.0;
	    		}
	    	}
	    }
		
        double VEHICLE_CAPACITY_precent;
        VEHICLE_CAPACITY_precent = 0;
	    for (int i = 0; i < (numberOfCities_ ); i++) {
	        int x ; 
	        int y ;
	        int x_last;

	        x = ((Permutation)solution.getDecisionVariables()[0]).vector_[i] ;   
	        if (i==0) {

	      	  weights_precent = new_node_v[x];
	        }
	        else {
	            x_last = ((Permutation)solution.getDecisionVariables()[0]).vector_[i-1] ; 
	            //System.out.println(if_start0[i-1]);
	            if ( (if_start0[i-1]- 48) == 1) {

	          	  for (int j = 0; j < Type_trucks; j++) {
	          		  if(route_hard[nroute]*(VEHICLE_VOLUME[j]*Relax_ratio)>weights_precent) {
	          			  VEHICLE_CAPACITY_precent = route_hard[nroute]*(VEHICLE_VOLUME[j]*Relax_ratio);
	          			  break;
	          		  }
	          		  VEHICLE_CAPACITY_precent =route_hard[nroute]* (VEHICLE_VOLUME[j]*Relax_ratio);
	          	  }

			      //System.out.println(weights_precent/VEHICLE_CAPACITY_precent);
//	          	  n_truck_used += 1;
		      	  //System.out.println(VEHICLE_CAPACITY_precent);
	          	  if (weights_precent > VEHICLE_CAPACITY_precent) {
	          		  w_const += weights_precent - VEHICLE_CAPACITY_precent;
	          	  }
	          	  w_all += weights_precent;
	          	  weights_precent = new_node_v[x];
	          	  nroute +=1;
	            }
	            else {

	          	  weights_precent += new_node_v[x];
	            }
	        }
	        //System.out.println(weights_precent);
	        if (i==(numberOfCities_-1 )) {
	          w_all += weights_precent;
	      	  for (int j = 0; j < Type_trucks; j++) {
	      		  if(route_hard[nroute]*(VEHICLE_VOLUME[j]*Relax_ratio)>weights_precent) {
	      			  VEHICLE_CAPACITY_precent = route_hard[nroute]*(VEHICLE_VOLUME[j]*Relax_ratio);
	      			  break;
	      		  }
	      		  VEHICLE_CAPACITY_precent = route_hard[nroute]*(VEHICLE_VOLUME[j]*Relax_ratio);
	      	  }

	      	  
		      //System.out.println(weights_precent/VEHICLE_CAPACITY_precent);
//	      	  n_truck_used += 1;
	      	  //System.out.println(VEHICLE_CAPACITY_precent);
	      	  if (weights_precent > VEHICLE_CAPACITY_precent) {
	      		  w_const += weights_precent - VEHICLE_CAPACITY_precent;
	      	  }
	        }

	  //  cout << "I : " << i << ", x = " << x << ", y = " << y << endl ; 
	       // System.out.println(weights_precent);
	      }
	    
	    
	    int number = 0;
	    double total = 0.0;
 	    if (w_const > 0) {
	    	number += 1;
	    	total -= w_const;
	    }
 	    if (first_const > 0) {
	    	number += 1;
	    	total -= first_const;
	    }
 	    
	    solution.setOverallConstraintViolation(total);    
	    solution.setNumberOfViolatedConstraint(number);	    
	    
	  } // evaluateConstraints
  
  
  public void evaluateConstraints_final(Solution solution) throws JMException {
	    double fitness1   ;
	    double fitness2   ;
	    double weights_precent ;
	    int n_truck_used ;
	    double w_const ;
	    double w_all = 0.0;
	    int new_num_nodes = 0;

	    fitness1   = 0.0 ;
	    fitness2   = 0.0 ;
	    weights_precent = 0.0;
	    n_truck_used = 0;
	    w_const = 0;
	    int Type_trucks = VEHICLE_CAPACITY.length;
	    
		String if_start = solution.getDecisionVariables()[1].toString();
		char [] if_start0 = if_start.toCharArray();
		
	    new_node_no = new int [numberOfCities_];
	    new_node_v = new double[numberOfCities_];

	    int var = 2;
	    for (int i = 0; i < NUM_nodes; i++) {
	    	if  (nodes_v_w[i][0] > Split_minv) {
	    		String s = solution.getDecisionVariables()[var].toString() ; 
	    	    char [] s0 = s.toCharArray();
	    	    double s00 = 0;
	    	    for (int j=0;j<2;j++) {
	    	    	s00 += (s0[j]- 48)*2^j;
	    	    }
	    	    //new_node_coords[new_num_nodes] = node_coords[i];
	    	    new_node_no[new_num_nodes] = i+1;
	        	new_node_v[new_num_nodes] = nodes_v_w[i][0]*(s00/Split_precision);
	        	new_num_nodes += 1; 
	    	    //new_node_coords[new_num_nodes] = node_coords[i];
	        	new_node_no[new_num_nodes] = i+1;
	        	new_node_v[new_num_nodes] = nodes_v_w[i][0]*(1.0-s00/Split_precision);
	        	new_num_nodes += 1; 
	    		
	    		var += 1;
	    	}
	    	else {
	    		//new_node_coords[new_num_nodes] = node_coords[i];
	    		new_node_no[new_num_nodes] = i+1;
	    		new_node_v[new_num_nodes] = nodes_v_w[i][0];    	
	    		new_num_nodes += 1; 	
	    	}

	    } 
		
      double VEHICLE_CAPACITY_precent;
      VEHICLE_CAPACITY_precent = 0;
	    for (int i = 0; i < (numberOfCities_ ); i++) {
	        int x ; 
	        int y ;
	        int x_last;

	        x = ((Permutation)solution.getDecisionVariables()[0]).vector_[i] ;   
	        if (i==0) {
	      	  fitness1 += distance2_start_end(0,x);
	      	  weights_precent = new_node_v[x];
	        }
	        else {
	            x_last = ((Permutation)solution.getDecisionVariables()[0]).vector_[i-1] ; 
	            //System.out.println(if_start0[i-1]);
	            if ( (if_start0[i-1]- 48) == 1) {
	          	  fitness1 += distance2_start_end(NUM_nodes+1,x_last);
	          	  fitness1 += distance2_start_end(0,x);
	          	  for (int j = 0; j < Type_trucks; j++) {
	          		  if((VEHICLE_VOLUME[j]*Relax_ratio)>weights_precent) {
	          			  VEHICLE_CAPACITY_precent = (VEHICLE_VOLUME[j]*Relax_ratio);
	          			  break;
	          		  }
	          		  VEHICLE_CAPACITY_precent = (VEHICLE_VOLUME[j]*Relax_ratio);
	          	  }
	          	  fitness2 += weights_precent/VEHICLE_CAPACITY_precent;
			      //System.out.println(weights_precent/VEHICLE_CAPACITY_precent);
	          	  n_truck_used += 1;
		      	  //System.out.println(VEHICLE_CAPACITY_precent);
	          	  if (weights_precent > VEHICLE_CAPACITY_precent) {
	          		  w_const += 1;
	          	  }
	          	  w_all += weights_precent;
	          	  weights_precent = new_node_v[x];
	            }
	            else {
	          	  fitness1 += distance2(x,x_last);
	          	  weights_precent += new_node_v[x];
	            }
	        }
	        //System.out.println(weights_precent);
	        if (i==(numberOfCities_-1 )) {
	          w_all += weights_precent;
	      	  for (int j = 0; j < Type_trucks; j++) {
	      		  if((VEHICLE_VOLUME[j]*Relax_ratio)>weights_precent) {
	      			  VEHICLE_CAPACITY_precent = (VEHICLE_VOLUME[j]*Relax_ratio);
	      			  break;
	      		  }
	      		  VEHICLE_CAPACITY_precent = (VEHICLE_VOLUME[j]*Relax_ratio);
	      	  }
	      	  fitness1 += distance2_start_end(NUM_nodes+1,x);
	      	  fitness2 += weights_precent/VEHICLE_CAPACITY_precent;
	      	  
		      //System.out.println(weights_precent/VEHICLE_CAPACITY_precent);
	      	  n_truck_used += 1;
	      	  //System.out.println(VEHICLE_CAPACITY_precent);
	      	  if (weights_precent > VEHICLE_CAPACITY_precent) {
	      		  w_const += 1;
	      	  }
	        }

	  //  cout << "I : " << i << ", x = " << x << ", y = " << y << endl ; 
	       // System.out.println(weights_precent);
	      }
	    
	    fitness1 = fitness1/10000;
	    fitness2 = 1-fitness2/n_truck_used;
	    

//    	System.out.println(n_truck_used);
//		for(int i=0;i<VEHICLE_VOLUME.length;i++) {
//	    	System.out.println(VEHICLE_VOLUME[i]);
//		  }
//    	System.out.println(VEHICLE_CAPACITY_precent);
	    System.out.println(w_const);
    	System.out.println(fitness1);
    	System.out.println(fitness2);
//    	
//    	System.out.println("check");

	    
	    
	    int number = 0;
	    double total = 0.0;
	    if (w_const > 0) {
	    	number = 1;
	    	total = -w_const;
	    }
	    solution.setOverallConstraintViolation(total);    
	    solution.setNumberOfViolatedConstraint(number);	    
	    
	  } // evaluateConstraints  
/* 
  public void evaluateConstraints_final (Solution solution) throws JMException {
	    double fitness1   ;
	    double fitness2   ;
	    double weights_precent ;
	    int n_truck_used ;
	    double w_const ;
	    double w_all = 0.0;
	    int new_num_nodes = 0;

	    fitness1   = 0.0 ;
	    fitness2   = 0.0 ;
	    weights_precent = 0.0;
	    n_truck_used = 0;
	    w_const = 0;
	    int Type_trucks = VEHICLE_CAPACITY.length;
	    
		String if_start = solution.getDecisionVariables()[1].toString();
		char [] if_start0 = if_start.toCharArray();
		
		
	    new_node_coords = new double[numberOfCities_][2];
	    new_node_weights = new double[numberOfCities_];

	    int var = 2;
	    for (int i = 0; i < NUM_nodes; i++) {
	    	if  (node_weights[i] > Split_minweight) {
	    		String s = solution.getDecisionVariables()[var].toString() ; 
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
		

	    for (int i = 0; i < (numberOfCities_ ); i++) {
	        int x ; 
	        int y ;
	        int x_last;
	        double VEHICLE_CAPACITY_precent;
	        VEHICLE_CAPACITY_precent = 0;
      
	               
	        
	        x = ((Permutation)solution.getDecisionVariables()[0]).vector_[i] ;   
	        if (i==0) {
	      	  fitness1 += distance2_start_end(start_coords,x);
	      	  weights_precent = new_node_weights[x];
	        }
	        else {
	            x_last = ((Permutation)solution.getDecisionVariables()[0]).vector_[i-1] ; 
	            //System.out.println(if_start0[i-1]);
	            if ( (if_start0[i-1]- 48) == 1) {
	          	  fitness1 += distance2_start_end(end_coords,x_last);
	          	  fitness1 += distance2_start_end(start_coords,x);
	          	  for (int j = 0; j < Type_trucks; j++) {
	          		  if(VEHICLE_CAPACITY[j]>weights_precent) {
	          			  VEHICLE_CAPACITY_precent = VEHICLE_CAPACITY[j];
	          			  break;
	          		  }
	          		  VEHICLE_CAPACITY_precent = VEHICLE_CAPACITY[j];
	          	  }
	          	  fitness2 += weights_precent/VEHICLE_CAPACITY_precent;
	          	  n_truck_used += 1;
		      	  //System.out.println(VEHICLE_CAPACITY_precent);
	          	  if (weights_precent > VEHICLE_CAPACITY_precent) {
	          		  w_const += weights_precent - VEHICLE_CAPACITY_precent;
	          	  }
	          	  w_all += weights_precent;
	          	  weights_precent = new_node_weights[x];
	            }
	            else {
	          	  fitness1 += distance2(x,x_last);
	          	  weights_precent += new_node_weights[x];
	            }
	        }
	        //System.out.println(weights_precent);
	        if (i==(numberOfCities_-1 )) {
	          w_all += weights_precent;
	      	  for (int j = 0; j < Type_trucks; j++) {
	      		  if(VEHICLE_CAPACITY[j]>weights_precent) {
	      			  VEHICLE_CAPACITY_precent = VEHICLE_CAPACITY[j];
	      			  break;
	      		  }
	      		  VEHICLE_CAPACITY_precent = VEHICLE_CAPACITY[j];
	      	  }
	      	  fitness1 += distance2_start_end(end_coords,x);
	      	  fitness2 += weights_precent/VEHICLE_CAPACITY_precent;
	      	  n_truck_used += 1;
	      	  //System.out.println(VEHICLE_CAPACITY_precent);
	      	  if (weights_precent > VEHICLE_CAPACITY_precent) {
	      		  w_const += weights_precent - VEHICLE_CAPACITY_precent;
	      	  }
	        }

	  //  cout << "I : " << i << ", x = " << x << ", y = " << y << endl ; 
	       // System.out.println(weights_precent);
	      }
	    
	    int number = 0;
	    double total = 0.0;
	    if (w_const > 0) {
	    	number = 1;
	    	total = -w_const;
	    }
	    //solution.setOverallConstraintViolation(total);    
	    //solution.setNumberOfViolatedConstraint(number);	    
	    System.out.println(w_const);
	  } // evaluateConstraints
*/
  private double distance2(int start_c,int end_c) {
		//double x1 = new_node_coords[start_c][0];
		//double y1 = new_node_coords[start_c][1];
		//double x2 = new_node_coords[end_c][0];
		//double y2 = new_node_coords[end_c][1];
		String twoPlatform;
		twoPlatform = String.valueOf(new_node_no[start_c])+'+'+String.valueOf(new_node_no[end_c]);
	    //return Math.sqrt(Math.pow((x1-x2),2) + Math.pow((y1-y2),2));
		//System.out.println(twoPlatform);
		//HashMap<String, Double> Distancescheck = Distances;
	    return Distances.get(twoPlatform);
	  }
  private double distance2_start_end(int start_end,int mid) {
		String twoPlatform;
		if (start_end==0) {
			twoPlatform = String.valueOf(start_end)+'+'+String.valueOf(new_node_no[mid]);
		}
		else {
			twoPlatform = String.valueOf(new_node_no[mid])+'+'+String.valueOf(start_end);
		}
		//System.out.println(twoPlatform);
		//HashMap<String, Double> Distancescheck = Distances;
	    //return Math.sqrt(Math.pow((x1-x2),2) + Math.pow((y1-y2),2));
	    return Distances.get(twoPlatform);
	  }

}


