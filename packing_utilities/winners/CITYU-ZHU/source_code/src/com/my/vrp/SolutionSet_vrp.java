package com.my.vrp;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

public class SolutionSet_vrp implements Serializable {
	protected final List<Solution_vrp> solutionList_;
	  /** 
	   * Maximum size of the solution set 
	   */
	  private int capacity_ = 0; 
	public SolutionSet_vrp() {
		solutionList_ = new ArrayList<Solution_vrp>();
	}
	
	public SolutionSet_vrp(int maximumSize) {
		solutionList_ = new ArrayList<Solution_vrp>();
		capacity_ = maximumSize;
	}
	
	public boolean add(Solution_vrp solution) {
		solutionList_.add(solution);
		return true;
	}
	
	public boolean add(int index, Solution_vrp solution) {
		solutionList_.add(index, solution);
		return true;
	}
	/**
	 * Replaces a solution by a new one
	 * 
	 * @param position
	 *            The position of the solution to replace
	 * @param solution
	 *            The new solution
	 */
	public void replace(int position, Solution_vrp solution) {
		if (position > this.solutionList_.size()) {
			solutionList_.add(solution);
		} // if
		solutionList_.remove(position);
		solutionList_.add(position, solution);
	} // replace
	
	public Solution_vrp get(int i) {
		if(i>=solutionList_.size()) {
			throw new IndexOutOfBoundsException("Index out of Bound "+i);
		}
		return solutionList_.get(i);
	}
	
	/**
	 * Returns the number of solutions in the SolutionSet.
	 * 
	 * @return The size of the SolutionSet.
	 */
	public int size() {
		return solutionList_.size();
	} // size
	
//	public void sort(Comparator comparator) {
//		if(comparator == null) {
//			System.out.println("No criterium for comparing exist.");
//			return;
//		}
//		Collections.sort(solutionList_,comparator);
//	}
	
	/** 
	   * Empties the SolutionSet
	   */
	  public void clear(){
	    solutionList_.clear();
	  } // clear
	    
	  /** 
	   * Deletes the <code>Solution</code> at position i in the set.
	   * @param i The position of the solution to remove.
	   */
	  public void remove(int i){        
	    if (i > solutionList_.size()-1) {            
//	      Configuration.logger_.severe("Size is: "+this.size());
	    } // if
	    solutionList_.remove(i);    
	  } // remove
	  private int dominance_compare(Solution_vrp p, Solution_vrp q) {
//	    	if(p.getF1()==q.getF1()&&p.getF2()==q.getF2()) {
//	    		return 0;
//	    	}else 
    		if(p.getF1()<=q.getF1()&&p.getF2()<=q.getF2()) {
	    		return 1;
	    	}else if(q.getF1()<=p.getF1()&&q.getF2()<=p.getF2()) {
	    		return -1;
	    	}
	    	return 0;
	    }
	  public void removeDomintes() {
		  int flagDominate;
		  boolean change=true;
		  do {
			  change=false;
		  for(int p=0;p<solutionList_.size();p++) {
			  for(int q=0;q<solutionList_.size()&&q!=p;q++) {
				  flagDominate = dominance_compare(solutionList_.get(p),solutionList_.get(q));
				  if(flagDominate == -1) {
//	        			iDominate[p].add(q);// q dominate p
					  solutionList_.remove(p);
					  change=true;
					  break;
	        		}
	        		else if (flagDominate == 1) {
//	        			iDominate[q].add(p);// p dominate q
	        			solutionList_.remove(q);
	        			change=true;
	        			break;
	        		}
			  }
			  if(change) break;
		  }
		  }while(change);
	  }
	
	public double get2DHV(Double [] idealNadir) {
		if(solutionList_.size()<1) return 0.0;
		if(idealNadir.length<4) {return -1;}//idealNadir必须包含4个数。
		double minF1 = idealNadir[0],minF2 = idealNadir[1],maxF1=idealNadir[2],maxF2=idealNadir[3];
		double hv = 0.0;
		//1. normalize this solutions
		double [][] f = new double[solutionList_.size()][2];//the normalized objective values
		for(int i=0;i<solutionList_.size();i++) {
			f[i][0] = (solutionList_.get(i).getF1()-minF1)/(maxF1-minF1);
			f[i][1] = (solutionList_.get(i).getF2()-minF2)/(maxF2-minF2);
		}
		
		//2. sort this solutionSet
//		Arrays.sort(f);
		Arrays.sort(f, new java.util.Comparator<double[]>() {
		    public int compare(double[] a, double[] b) {
		    	if(a[0]<b[0]) {
		    		return -1;
		    	}else if(a[0]>b[0]) {
		    		return 1;
		    	}else {
		    		if(a[1]<b[1]) {
		    			return -1;
		    		}else if(a[1]>b[1]) {
		    			return 1;
		    		}else {
		    			return 0;
		    		}
		    	}
//		        return Double.compare(a[1], b[1]);
		    }
		});
		//筛选非支配解。
//		ArrayList [] nd_f= new ArrayList[2];
//		nd_f[0] = new ArrayList<Double>();
//		nd_f[1] = new ArrayList<Double>();
//		nd_f[0].add(f[0][0]);
//		nd_f[1].add(f[0][1]);
//		double former0 = f[0][0], former1 = f[0][1];
//		for(int i=1;i<solutionList_.size();i++) {
//			if(!(f[i][0]>=former0&&f[i][1]>=former1)) {//
//				nd_f[0].add(f[i][0]);
//				nd_f[1].add(f[i][1]);
//				former0=f[i][0];
//				former1=f[i][1];
//			}
//		}
		hv = Math.abs((f[0][0]-1.2)*(f[0][1]-1.2));
		
		for(int i=1;i<solutionList_.size();i++) {
			//hv += Math.abs(((double)nd_f[0].get(i)-1.2)*((double)nd_f[1].get(i)-f[i-1][1]));
			hv += Math.abs(((double)f[i][0]-1.2)*(f[i][1]-f[i-1][1]));
		}
		
		return hv;
	}
	
	
	
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
	                        
	      for (int i = 0; i < solutionList_.size(); i++) {
	        //if (this.vector[i].getFitness()<1.0) {
	        bw.write(solutionList_.get(i).toString());
	        bw.newLine();
	        //}
	      }
	      
	      /* Close the file */
	      bw.close();
	    }catch (IOException e) {
//	      Configuration.logger_.severe("Error acceding to the file");
	      e.printStackTrace();
	    }
	  } // printObjectivesToFile
	  
	  
	  /**
	   * delete the same solution 
	   * */
	  public void Suppress() {
//			int decisionnum = 2;//solutionList_.get(0).getNumberOfObjectives();
			double diff;
			for (int k = 0; k < solutionList_.size(); k++) {
				for (int l = k + 1; l < solutionList_.size(); l++) {
//					int m = 0;
					//for (m = 0; m < decisionnum; m++) {
					diff=Math.abs(solutionList_.get(k).getF1()-solutionList_.get(l).getF1())+Math.abs(solutionList_.get(k).getF2()-solutionList_.get(l).getF2());
//						if(diff<0)
//							diff=-diff;
					if(diff<0.0001){
						solutionList_.remove(l);
						l--;
	    			}
					//}
//					if (m == decisionnum) {
//						solutionList_.remove(l);
//						l--;
//					}
				}
			}
	  }
}
