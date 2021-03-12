package com.my.vrp.operator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import com.my.vrp.Box;
import com.my.vrp.Solution_vrp;

//import static com.my.vrp.utils.DistanceSolution.caculateDistanceSolution;
public class TwoSwap extends Move{

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
		
		//找超过3个节点的（除了起始和结束）的路径a。
		int a;
		Random random = new Random();
		int iter=0;
		do {
			a = random.nextInt(solution.getRoutes().size());
			iter++;
		}while(solution.getRoutes().get(a).getNodes().size()-2<3&&iter<1000);
		if(iter>=1000) return false;
		//在这条路径中随机选择2个不同的点x,y。
		int x;
		int y;
		do {
			x = random.nextInt(solution.getRoutes().get(a).getNodes().size()-2)+1;
			y = random.nextInt(solution.getRoutes().get(a).getNodes().size()-2)+1;
		}while(x==y||solution.getRoutes().get(a).getNodes().get(x).isMustFirst()||solution.getRoutes().get(a).getNodes().get(y).isMustFirst());
		//注意这两个节点不可以是mustFirst的。
		
		int i = solution.getRoutes().get(a).getNodes().get(x).getPlatformID();//客户x的编号
		int j = solution.getRoutes().get(a).getNodes().get(y).getPlatformID();//客户y的编号
		Collections.swap(solution.getRoutes().get(a).getNodes(), x, y);
		
		
		
		
//		//转换route a里面的carriage.
//		ArrayList<Box> Boxes_old = solution.getRoutes().get(a).getCarriage().getBoxes();
////		String platformx=solution.getRoutes().get(a).getNodes().get(x).getPlatform();
////		String platformy=solution.getRoutes().get(a).getNodes().get(y).getPlatform();
//		int beginx=0,beginy=0;
//		boolean beginxFlag=false,beginyFlag=false;
//		int n_x=0,n_y=0;//total boxes belong to platformx and platformy
//		for(int boxi=0;boxi<Boxes_old.size();boxi++) {
//			//遍历所有的boxes,1). 找到node x的起始和结束的位置。2). 找到node y的起始和结束的位置。
//			Box curr_box = Boxes_old.get(boxi);
//			if(curr_box.getPlatformid()==i) {
//				if(beginxFlag==false) {beginx=boxi;beginxFlag=true;}
//				n_x++;
//			}
//			if(curr_box.getPlatformid()==j) {
//				if(beginyFlag==false) {beginy=boxi;beginyFlag=true;}
//				n_y++;
//			}
//		}
//		//0->Boxes(min(x,y))->Boxes(max(x,y))->end
//		//0->Boxes(max(x,y))->Boxes(min(x,y))->end
//		//0->beginx->beginx+n_x->beginy->beginy+n_y->end
//		//第一段：0->beginx-1
//		//第二段：beginy->beginy+n_y-1
//		//第三段：beginx+n_x-1+1->beginy-1
//		//第四段：beginx->beginx+n_x-1
//		//第五段：beginy+n_y-1+1->end
////		int minxy=Math.min(beginx,beginy);
////		int maxxy=Math.max(beginx, beginy);
//		ArrayList<Box> Boxes_new = new ArrayList<Box>();
//		int minxy,minn,maxxy,maxn;
//		if(beginx<beginy) {
//			minxy=beginx;minn=n_x;
//			maxxy=beginy;maxn=n_y;
//		}else {
//			minxy=beginy;minn=n_y;
//			maxxy=beginx;maxn=n_x;
//		}
//		Boxes_new.addAll(Boxes_old.subList(0, minxy-1));//0
//		Boxes_new.addAll(Boxes_old.subList(maxxy, maxxy+maxn-1));
//		Boxes_new.addAll(Boxes_old.subList(minxy+minn, maxxy-1));
//		Boxes_new.addAll(Boxes_old.subList(minxy, minxy+minn-1));
//		Boxes_new.addAll(Boxes_old.subList(maxxy+maxn, Boxes_old.size()-1));
//		solution.getRoutes().get(a).getCarriage().setBoxes(Boxes_new);
//		//=============================================
		
		
		
		
		solution.setFitness(solution.caculateDistanceSolution(solution.getRoutes()));//用原来的装载情况来进行评价的？？？？
		this.setSolution(solution);
		int[] movePattern= new int[] {i,j};
		this.setMovePattern(movePattern);
		this.setName("twoSwap");
		this.getRouteIdx().add(a);
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
