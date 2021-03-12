package com.my.vrp.operator;


import java.util.Random;


import com.my.vrp.Solution_vrp;

//import static com.my.vrp.utils.DistanceSolution.caculateDistanceSolution;
public class MoveClientx extends Move{

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
		//找两条不同的路径a,b
		//a的节点数是3个，也就是只包含一个解点的路径，（出了其实和结束节点）
		int a;
		int b;
		do {
			a = random.nextInt(solution.getRoutes().size());
			b = random.nextInt(solution.getRoutes().size());
		}while(solution.getRoutes().get(a).getNodes().size()==3||a==b);
		
		//在a,b中随机选择一个节点x,y（除了起始节点）
		int x = random.nextInt(solution.getRoutes().get(a).getNodes().size()-2)+1;
		int y = random.nextInt(solution.getRoutes().get(b).getNodes().size()-2)+1;
		//
		int i = solution.getRoutes().get(a).getNodes().get(x).getPlatformID();
		int j = solution.getRoutes().get(b).getNodes().get(y).getPlatformID();
		//在b的y节点上加入a的x节点。
		solution.getRoutes().get(b).getNodes().add(y, solution.getRoutes().get(a).getNodes().get(x));//把a路径的结点x插入到b路径的y位置。
		solution.getRoutes().get(a).getNodes().remove(x);
		//重新计算这个解的距离。
		solution.setFitness(solution.caculateDistanceSolution(solution.getRoutes()));
		this.setSolution(solution);
		//可复用代码标记，今后试图封装为方法
		int[] movePattern= new int[] {i,j};// 改变了平台i,j,// i是a路径取下来要插入j路径的结点
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
