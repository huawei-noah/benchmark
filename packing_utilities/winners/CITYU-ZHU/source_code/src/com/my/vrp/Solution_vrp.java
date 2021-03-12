package com.my.vrp;

//import static com.my.vrp.utils.CaculateDistance.caculateDistance;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map;

/**
 * 算法的解类
 * distanceSolution:距离解<br/>
 * routeSolution:路线解<br/>
 * @author dell
 *
 */
public class Solution_vrp implements Cloneable{
	private boolean isFeasible;
	private double fitness;//保存这个解的质量，相当于fitness
	private double f1;//1-\sum_1^n loading_rate_i/n
	private double f2;//distance.
	private int rank_ ;// Stores the so called rank of the solution. Used in NSGA-II
	  /**
	   * Stores the crowding distance of the the solution in a 
	   * <code>SolutionSet</code>. Used in NSGA-II.
	   */
	  private double crowdingDistance_ ; 
	public Map<String,Double> distanceMap;
	private ArrayList<Route> routes;//这个解的所有路径，包含经过的节点，以及需要装载的boxes
	
	public Solution_vrp() {
		routes = new ArrayList<Route>();
	}
	
	public Solution_vrp(Solution_vrp s) {
		isFeasible = s.isFeasible();
		fitness = s.getFitness();
		f1 = s.getF1();
		f2 = s.getF2();
		distanceMap = s.distanceMap;//这里要注意删除s的时候，是否会删除当前的distanceMap
		Iterator<Route> iterator = s.routes.iterator();
		this.routes = new ArrayList<Route>();
		while(iterator.hasNext()) this.routes.add(new Route(iterator.next()));
	}
	
	
	public void updateSolution(Solution_vrp solution) {
		this.fitness = solution.fitness;
		this.routes = solution.routes;
	}
	//get set for fitness
	public double getFitness() {
		return fitness;
	}
	public void setFitness(double fitness) {
		this.fitness = fitness;
	}
	//get set for routes
	public ArrayList<Route> getRoutes() {
		return routes;
	}
	public void setRoutes(ArrayList<Route> routes) {
		this.routes = routes;
	}
//	@Override
//	public String toString() {
//		return "Solution [distanceSolution=" + distanceSolution + ", routesSolution=" + routesSolution + "]";
//	}
//	public void showSolution(boolean finalsolution) {
//		System.out.println("Routes：");
//		for (Route route : this.routes) {
//			if(finalsolution) {
//			//打印这条路径//这条路径所有的boxes及其坐标。
//			System.out.print(""+route.getId()+" :Vehicle(l"+route.getCarriage().getLength()+",w"+route.getCarriage().getWidth()+",h"+route.getCarriage().getHeight()+",capacity:"+route.getCarriage().getCapacity()+") ");
////			for (Node node : route.getNodes()) {
//			//xyzlwh
////			System.out.print(route.getNodes().getPlatformID());
//			for(int i=0;i<route.getCarriage().getBoxes().size();i++)
//			System.out.print(
//					"p"+route.getCarriage().getBoxes().get(i).getPlatformid()+"(x"+route.getCarriage().getBoxes().get(i).getXCoor()+
//					 ",y"+route.getCarriage().getBoxes().get(i).getYCoor()+
//					 ",z"+route.getCarriage().getBoxes().get(i).getZCoor()+
//					 ",l"+route.getCarriage().getBoxes().get(i).getLength()+
//					 ",w"+route.getCarriage().getBoxes().get(i).getWidth()+
//					 ",h"+route.getCarriage().getBoxes().get(i).getHeight()+
//					")");
//			System.out.print("->");
////			}
//			}else {
//				//打印这条路径
//				System.out.print(""+route.getId()+" :\t");
//				for (Node node : route.getNodes()) {
//					//xyzlwh
//					System.out.print(node.getPlatformID()+"->");
//				}
//			}
//			System.out.println();
//		}
//		System.out.println("Distance：");
//		System.out.println(this.fitness);
//	}
	@Override
	public Object clone() throws CloneNotSupportedException {
		// TODO Auto-generated method stub
		Solution_vrp solution = null;
		try {
			solution = (Solution_vrp)super.clone();
		}catch(CloneNotSupportedException e) {
			e.printStackTrace();
		}
		//其他都一样，就是routes是空的吗？
		ArrayList<Route> routes = new ArrayList<Route>();
		Iterator<Route> iterator = this.routes.iterator();
		while(iterator.hasNext()) {
			routes.add((Route) iterator.next().clone());
		}
		solution.setRoutes(routes);
		return solution;
	}
	
	public boolean isFeasible() {
		return isFeasible;
	}
	public void setFeasible(boolean isFeasible) {
		this.isFeasible = isFeasible;
	}
	public void evaluation() {
		double loading_rate=0.0;
		for(Route route : this.routes) {
			double load_volume=0.0,load_weight=0.0;
			for(Box box : route.getBoxes()) {
				load_volume=load_volume+box.getHeight()*box.getLength()*box.getWidth();
				load_weight=load_weight+box.getWeight();
			}
			double V = route.getCarriage().getHeight()*route.getCarriage().getWidth()*route.getCarriage().getLength();
			double v_rate=load_volume/V;
//			if(Math.abs(route.getLoadWeight()-load_weight)>0.0001) 
//				System.out.println("error:this load weight:"+load_weight+" not equal to origin load weight:"+route.getLoadWeight());
			double w_rate=load_weight/route.getCarriage().getCapacity();
			loading_rate=loading_rate+Math.max(v_rate,w_rate);
		}
		loading_rate=loading_rate/this.routes.size();
		this.setF1(1.0-loading_rate);
		double distance = caculateDistanceSolution(this.getRoutes());
//		for (Route route : this.getRoutes()) {
//			for(int i1=0;i1<route.getNodes().size()-1;i1++) {//
//				String twoPlatform = String.valueOf(route.getNodes().get(i1).getPlatformID())+'+'+String.valueOf(route.getNodes().get(i1+1).getPlatformID());
//				distance+=distanceMap.get(twoPlatform);//caculateDistance(, );
//			}
//		}
//		if(distance!=this.fitness) System.out.println("why the distance not equal to fitness? is it a infeasible solution?");
		this.f2=distance;
	}
	/**
	 * 用于求带有惩罚的距离的方法
	 * @param solution
	 * @return
	 */
//		public static double caculateDistanceSolution(Solution solution) {
////					ArrayList<Carriage> carriages = new ArrayList<Carriage>();
////					for(int p=0;p<solution.getRoutesSolution().size();p++) {
////						Carriage carriage = new Carriage();
////						carriage.setLength(INF);
////						LoadConfirm loadConfirm = new LoadConfirm();
////						LinkedList<Node> nodes = new LinkedList<Node>();
////						for(int q=1;q<solution.getRoutesSolution().get(p).getNodes().size()-1;q++) {
////							nodes.add(solution.getRoutesSolution().get(p).getNodes().get(q));
////						}
////						loadConfirm.loadConfirm(nodes, carriage);
////						carriages.add(carriage);
////					}
//					double distance = caculateDistanceSolutionII(solution);
//					ExcessLength excessLength = new ExcessLength(solution.getRoutes());
//					ExcessWeight excessWeight = new ExcessWeight(solution.getRoutes());
//					double excessLengthPunish = excessLength.getExcessLength();
//					double excessWeightPulish = excessWeight.getExcessWeight();
////					ExcessTime excessTime = new ExcessTime(solution);
////					double excessTimePunish = excessTime.getExcessTime();
////					distance+=Math.pow(excessLengthPunish, 4)*B+Math.pow(excessWeightPulish, 4)*A
////							+Math.pow(excessTimePunish, 3)*C;
//					distance+=Math.pow(excessLengthPunish, 4)*B+Math.pow(excessWeightPulish, 4)*A;
//					return distance;
//		}
	/**
	 * 用于求不带惩罚的距离
	 * @param solution
	 * @return
	 */
	public double caculateDistanceSolution(Route route) {
		double distance = 0;
//		for (Route route : routes) {
		for(int i1=0;i1<route.getNodes().size()-1;i1++) {//
			String twoPlatform = String.valueOf(route.getNodes().get(i1).getPlatformID())+'+'+String.valueOf(route.getNodes().get(i1+1).getPlatformID());
//				distance+=distanceMap.get(twoPlatform);
//			if(this.distanceMap.containsKey(twoPlatform))
				distance += distanceMap.get(twoPlatform);//caculateDistance(route.getNodes().get(i1).getPlatformID(), route.getNodes().get(i1+1).getPlatformID());
//			else
//				System.out.println(twoPlatform);
//				System.exit(0);
		}
//		}
		return distance;
	}
	public double caculateDistanceSolution(ArrayList<Route> routes) {
		double distance = 0;
		for (Route route : routes) {
			distance+=caculateDistanceSolution(route);
//			for(int i1=0;i1<route.getNodes().size()-1;i1++) {//
//				String twoPlatform = String.valueOf(route.getNodes().get(i1).getPlatformID())+'+'+String.valueOf(route.getNodes().get(i1+1).getPlatformID());
////				distance+=distanceMap.get(twoPlatform);
//				distance+=this.distanceMap.get(twoPlatform);//caculateDistance(route.getNodes().get(i1).getPlatformID(), route.getNodes().get(i1+1).getPlatformID());
//			}
		}
		return distance;
	}
	public double getF1() {
		return f1;
	}
	public void setF1(double f1) {
		this.f1 = f1;
	}
	public double getF2() {
		return f2;
	}
	public void setF2(double f2) {
		this.f2 = f2;
	}

	public double getCrowdingDistance() {
		return crowdingDistance_;
	}

	public void setCrowdingDistance(double crowdingDistance_) {
		this.crowdingDistance_ = crowdingDistance_;
	}
	/** 
	   * Returns a string representing the solution.
	   * @return The string.
	   */
	  public String toString() {
	    String aux="";
//	    for (int i = 0; i < this.numberOfObjectives_; i++)
	      aux = aux + this.f1 + " ";
	      aux = aux + this.f2 + " ";
	    return aux;
	  } // toString

	public int getRank() {
		return rank_;
	}

	public void setRank(int rank_) {
		this.rank_ = rank_;
	}
}
