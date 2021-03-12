package com.my.vrp.operator;

import java.util.Random;

import com.my.vrp.Node;
import com.my.vrp.Solution_vrp;
//import static com.my.vrp.utils.DistanceSolution.caculateDistanceSolution;

public class TwoOpt extends Move{

	@Override
	public boolean fieldTransformation(Solution_vrp solution1) {
		// TODO Auto-generated method stub
		Solution_vrp solution = null;
		try {
			solution = (Solution_vrp)solution1.clone();
		} catch (CloneNotSupportedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		int a;
		Random random = new Random();
		int iter=0;
		do {//选择一条>3个节点的路径a
			a = random.nextInt(solution.getRoutes().size());
			iter=iter+1;
		}while(solution.getRoutes().get(a).getNodes().size()-2<3&&iter<1000);
		if(iter>=1000) return false;
		int x;
		int y;
		iter=0;
		do {//从该路径中选择2个节点。
			x = random.nextInt(solution.getRoutes().get(a).getNodes().size()-2)+1;
			y = random.nextInt(solution.getRoutes().get(a).getNodes().size()-2)+1;
			iter=iter+1;
		}while((x==y||(x==1&&y==solution.getRoutes().get(a).getNodes().size()-2)
				||(y==1&&x==solution.getRoutes().get(a).getNodes().size()-2))&&iter<1000);
		if(iter>=1000) return false;
		int i = solution.getRoutes().get(a).getNodes().get(x).getPlatformID();
		int j = solution.getRoutes().get(a).getNodes().get(y).getPlatformID();
		int z=(x>y)? x-y:y-x;
		if(x<y) {
			for(int p=0;p<=z/2;p++) {
				Node temp = null;
				try {
					temp = (Node) solution.getRoutes().get(a).getNodes().get(x+p).clone();
				} catch (CloneNotSupportedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				try {
					solution.getRoutes().get(a).getNodes().
					set(x+p, (Node) solution.getRoutes().get(a).getNodes().get(y-p).clone());
				} catch (CloneNotSupportedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} 
				solution.getRoutes().get(a).getNodes().set(y-p, temp);
			}	
		}
		else {
			for(int p=0;p<=z/2;p++) {
				Node temp = null;
				try {
					temp = (Node) solution.getRoutes().get(a).getNodes().get(y+p).clone();
				} catch (CloneNotSupportedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				try {
					solution.getRoutes().get(a).getNodes().
					set(y+p, (Node) solution.getRoutes().get(a).getNodes().get(x-p).clone());
				} catch (CloneNotSupportedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} 
				solution.getRoutes().get(a).getNodes().set(x-p, temp);
			}
		}
		solution.setFitness(solution.caculateDistanceSolution(solution.getRoutes()));
		this.setSolution(solution);
		int[] movePattern = new int[] {i,j};
		this.setMovePattern(movePattern);
		this.setName("twoOpt");
		return true;
	}
	@Override
	public int compareTo(Object o) {
		// TODO Auto-generated method stub
		Move move = (Move) o;
		if(this.getSolution().getFitness()>move.getSolution().getFitness())
			return 1;
		else if(this.getSolution().getFitness()==move.getSolution().getFitness())
			return 0;
		else
			return -1;
	}
}
