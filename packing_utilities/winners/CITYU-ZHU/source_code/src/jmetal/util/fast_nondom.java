package jmetal.util;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.logging.Handler;

public class fast_nondom {

	private int check_dominate(double[] a, double[] b) {
		int flag;

		flag = 0;

		
		for (int i = 0; i < a.length; i++) {
			if (a[i] > b[i]) {
				flag += 1;
			}
		}
		if (flag == a.length) {
			return 1;
		}
		flag = 0;
		int flag0 = 0;
		for (int i = 0; i < a.length; i++) {
			if (a[i] > b[i]) {
				flag += 1;
			}
			if (a[i] >= b[i]) {
				flag0 += 1;
			}
		}
		if (flag0 == a.length && flag > 0) {
			return 1;
		}

		return 0;

	}

    public HashMap fast_nondominated_sort(ArrayList<double[]> save_objs0) {
		int n = 0;
		HashMap output = new HashMap();
		ArrayList<Integer> no_PF = new ArrayList<Integer>();
		ArrayList<double[]> PF = new ArrayList<double[]>();
		
		for (int i = 0; i < save_objs0.size(); i++) {
			int nflag = 0;
			for (int j = 0; j < save_objs0.size(); j++) {
				if (j!=i) {
					int flag = check_dominate(save_objs0.get(i), save_objs0.get(j));
					nflag += flag;	
				}
			}
			
			if (nflag == 0) {
				 no_PF.add(i);
				 PF.add(save_objs0.get(i));
				 n += 1;
			} 

		}
		
        double[][] PF0 = new double[PF.size()][2];
        int[] no_PF0 = new int[PF.size()];
        int n_true = 1;
    	PF0[0] = PF.get(0);
    	no_PF0[0] = no_PF.get(0);
        for (int n1 = 0; n1<PF.size(); n1++) {
        	int flag = 0;
        	for (int n2 = 0; n2<n_true; n2++)
        		if (PF0[n2][0] == PF.get(n1)[0]&&PF0[n2][1] == PF.get(n1)[1]) {
        			flag = 1;
        		}
        	if (flag ==  0) {
        		PF0[n_true] = PF.get(n1);
        		no_PF0[n_true] = no_PF.get(n1);
        		n_true += 1;
        	}

        }
        
		output.put("PF",PF0);
		output.put("no_PF",no_PF0);
		output.put("n",n_true);

		return output;
    }
}