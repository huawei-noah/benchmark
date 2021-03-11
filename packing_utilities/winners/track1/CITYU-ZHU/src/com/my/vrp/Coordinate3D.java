package com.my.vrp;

public class Coordinate3D implements Comparable{
	private double[] coordinate;

	public double[] getCoordinate() {
		return coordinate;
	}

	public void setCoordinate(double[] coordinate) {
		this.coordinate = coordinate;
	}

	@Override
	public int compareTo(Object o) {
		// TODO Auto-generated method stub
		Coordinate3D c1 = (Coordinate3D) o;
		if(this.coordinate[0]==c1.coordinate[0]&&this.coordinate[1]==c1.coordinate[1])
			return 0;
		
		else
			return 1;
	}
	
	
}
