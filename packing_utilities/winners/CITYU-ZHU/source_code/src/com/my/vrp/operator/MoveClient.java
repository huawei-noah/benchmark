package com.my.vrp.operator;


import java.util.Random;

import com.my.vrp.Node;
import com.my.vrp.Solution_vrp;

//import static com.my.vrp.utils.DistanceSolution.caculateDistanceSolution;
public class MoveClient extends Move{

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
		Random random = new Random();
		int a;
		int j;
		do {
			a = random.nextInt(solution.getRoutes().size());
			j = random.nextInt(solution.getRoutes().size());
		}while(solution.getRoutes().get(a).getNodes().size()<=3||a==j);
		int x = random.nextInt(solution.getRoutes().get(a).getNodes().size()-2)+1;
		int y = random.nextInt(solution.getRoutes().get(j).getNodes().size()-2)+1;
		int i = solution.getRoutes().get(a).getNodes().get(x).getPlatformID();
		solution.getRoutes().get(j).getNodes().add(y, new Node(solution.getRoutes().get(a).getNodes().get(x)));//把a路径的结点插入到j路径
		solution.getRoutes().get(a).getNodes().remove(x);
		solution.setFitness(solution.caculateDistanceSolution(solution.getRoutes()));
		this.setSolution(solution);
		//可复用代码标记，今后试图封装为方法
		int[] movePattern= new int[] {i,j};//i是a路径取下来要插入j路径的结点
		this.setMovePattern(movePattern);
		this.setSolution(solution);
		this.setName("moveClient");
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
