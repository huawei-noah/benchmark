package com.my.vrp;
import java.util.*;
public class Ranking {
	/**
     * The <code>SolutionSet</code> to rank
     */
    private SolutionSet_vrp   solutionSet_ ;
    /**
     * An array containing all the fronts found during the search
     */
    private SolutionSet_vrp[] ranking_  ;
    
    
    
    public Ranking(SolutionSet_vrp solutionSet) {
    	solutionSet_ = solutionSet;
    	int[] dominateMe = new int[solutionSet_.size()];
    	List<Integer> [] iDominate = new List[solutionSet_.size()];
    	
    	List<Integer> [] front = new List[solutionSet_.size()+1];
    	
    	// flagDominate is an auxiliar variable
        int flagDominate;  
        
    	// Initialize the fronts 
        for (int i = 0; i < front.length; i++)
        	front[i] = new LinkedList<Integer>();
        
        for (int p=0;p<solutionSet_.size();p++) {
        	iDominate[p] = new LinkedList<Integer>();
        	dominateMe[0] = 0;
        }
        
        for (int p=0;p<solutionSet_.size()-1;p++) {
        	for (int q=p+1;q<solutionSet_.size();q++) {
        		flagDominate = dominance_compare(solutionSet.get(p),solutionSet.get(q));
        		if(flagDominate == -1) {
        			iDominate[p].add(q);// q dominate p
        		}
        		else if (flagDominate == 1) {
        			iDominate[q].add(p);// p dominate q
        		}
        	}
        	// If nobody dominates p, p belongs to the first front.
        }
        
        for (int p=0; p<solutionSet_.size(); p++) {
        	if(dominateMe[p] == 0) {
        		front[0].add(p);
//        		solutionSet.get(p).setRank(0);
        	}
        }
        //Obtain the rest of fronts
        int i=0;
        Iterator<Integer> it1, it2 ;//
        while (front[i].size()!=0) {
        	i++;
        	it1 = front[i-1].iterator();
        	while(it1.hasNext()) {
        		it2 = iDominate[it1.next().intValue()].iterator();
        		while (it2.hasNext()) {
        			int index = it2.next().intValue();
        			dominateMe[index]--;
        			if(dominateMe[index]==0) {
        				front[i].add(new Integer(index));
//        				solutionSet_.get(index).setRank(i); 
        			}
        		}
        	}
        }
        ranking_ = new SolutionSet_vrp[i];
        //0,1,2,....,i-1 are front, then i fronts
        for (int j = 0; j < i; j++) {
          ranking_[j] = new SolutionSet_vrp(front[j].size());
          it1 = front[j].iterator();
          while (it1.hasNext()) {
                    ranking_[j].add(solutionSet.get(it1.next().intValue()));       
          }
        }
    }//Ranking
    
    private int dominance_compare(Solution_vrp p, Solution_vrp q) {
    	if(p.getF1()==q.getF1()&&p.getF2()==q.getF2()) {
    		return 0;
    	}else if(p.getF1()<=q.getF1()&&p.getF2()<=q.getF2()) {
    		return 1;
    	}else if(q.getF1()<=p.getF1()&&q.getF2()<=p.getF2()) {
    		return -1;
    	}
    	return 0;
    }
    /**
     * Returns a <code>SolutionSet</code> containing the solutions of a given rank. 
     * @param rank The rank
     * @return Object representing the <code>SolutionSet</code>.
     */
    public SolutionSet_vrp getSubfront(int rank) {
      return ranking_[rank];
    } // getSubFront

    /** 
    * Returns the total number of subFronts founds.
    */
    public int getNumberOfSubfronts() {
      return ranking_.length;
    } // getNumberOfSubfronts
}
