package com.my.vrp.operator;



import java.util.LinkedList;
import java.util.Random;

import com.my.vrp.Node;
import com.my.vrp.Solution_vrp;
//import static com.my.vrp.utils.DistanceSolution.caculateDistanceSolution;
public class CrossOver extends Move{

	@Override
	public boolean fieldTransformation(Solution_vrp solution1) {
		// TODO Auto-generated method stub
		
		Solution_vrp solution = null;//经过crossover产生的解。
		try {
			solution = (Solution_vrp)solution1.clone();
		} catch (CloneNotSupportedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		//随机选择路径i,j
		Random random = new Random();
		int i,j;
		int iter=0;
		do {
			i = random.nextInt(solution.getRoutes().size());
			j = random.nextInt(solution.getRoutes().size());
			iter++;
		}while(i==j&&iter<1000);
		if(iter>=1000) return false;
		//随机在两条路径选择客户点a,b
		int a,b;
		a = random.nextInt(solution.getRoutes().get(i).getNodes().size()-2)+1;
		b = random.nextInt(solution.getRoutes().get(j).getNodes().size()-2)+1;
		
		int a1 = solution.getRoutes().get(i).getNodes().get(a).getPlatformID();//a1 is the id of this node, this is used for tabu.
		int b1 = solution.getRoutes().get(j).getNodes().get(b).getPlatformID();//b1 is the id of this node, this is used for tabu.
		LinkedList<Node> nodesi1 = new LinkedList<Node>();//route i 的前面一段节点（0-a)
		for(int p=0;p<=a;p++) {
			nodesi1.add(solution.getRoutes().get(i).getNodes().get(p));
		}
		LinkedList<Node> nodesi2 = new LinkedList<Node>();//route i 的后一段节点
		for(int p=a+1;p<solution.getRoutes().get(i).getNodes().size();p++) {
			nodesi2.add(solution.getRoutes().get(i).getNodes().get(p));
		}
		LinkedList<Node> nodesj1 = new LinkedList<Node>();//route j 的前一段节点
		for(int p=0;p<=b;p++) {
			nodesj1.add(solution.getRoutes().get(j).getNodes().get(p));
		}
		LinkedList<Node> nodesj2 = new LinkedList<Node>();//route j的后一段节点
		for(int p=b+1;p<solution.getRoutes().get(j).getNodes().size();p++) { 
			nodesj2.add(solution.getRoutes().get(j).getNodes().get(p));
		}
		//进行单点交叉操作。
		nodesi1.addAll(nodesj2);
		nodesj1.addAll(nodesi2);
		//
		solution.getRoutes().get(i).setNodes(nodesi1);
		solution.getRoutes().get(j).setNodes(nodesj1);
		//重新计算路径长度
		solution.setFitness(solution.caculateDistanceSolution(solution.getRoutes()));
		this.setSolution(solution);
//		this.setSolution(solution);
		
		int[] movePattern = new int[] {a1,b1};//platformIDs
		this.setMovePattern(movePattern);
		this.setName("crossOver");
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
