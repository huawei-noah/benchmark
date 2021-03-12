package com.my.vrp.param;

import java.util.ArrayList;

import com.my.vrp.Route;

/**
 * 用于节约算法的L序列
 * @author dell
 *
 */
@SuppressWarnings("rawtypes")
public class L implements Comparable{
	private double sij;//结合route i和j带来的路径上的提升。
	private double smn;
	//结合后的node顺序，
	private int i;//route i
	private int j;//route j
	private int m;
	private int n;
	private ArrayList<Integer> overlapIdx;//两条路径共同的节点。
	private Route route1;
	private Route route2;
	public double getSij() {
		return sij;
	}
	public void setSij(double sij) {
		this.sij = sij;
	}
	public int getI() {
		return i;
	}
	public void setI(int i) {
		this.i = i;
	}
	public int getJ() {
		return j;
	}
	public void setJ(int j) {
		this.j = j;
	}
	/**
	 * 按sij,smn从小到大的顺序。
	 */
	public int compareTo(Object o) {
		L l = (L) o;
		if(this.sij>l.sij)
			return -1;
		else if(this.sij<l.sij)
			return 1;
		else
			if(this.smn<l.smn)
				return -1;
			else if(this.smn>l.smn)
				return 1;
			else
				return 0;
				
	}
	@Override
	public String toString() {
		return "L [sij=" + sij + ", i=" + i + ", j=" + j + "]";
	}
	public ArrayList<Integer> getOverlapIdx() {
		return overlapIdx;
	}
	public void setOverlapIdx(ArrayList<Integer> overlapIdx) {
		this.overlapIdx = new ArrayList<Integer>();
		for(int idx:overlapIdx)
		this.overlapIdx.add(idx);
	}
	public void setOverlapIdx(int overlapIdx[]) {
		this.overlapIdx = new ArrayList<Integer>();
		for(int idx:overlapIdx)
		this.overlapIdx.add(idx);
	}
	public int getM() {
		return m;
	}
	public void setM(int m) {
		this.m = m;
	}
	public int getN() {
		return n;
	}
	public void setN(int n) {
		this.n = n;
	}
	public double getSmn() {
		return smn;
	}
	public void setSmn(double smn) {
		this.smn = smn;
	}
	public Route getRoute1() {
		return route1;
	}
	public void setRoute1(Route route1) {
		this.route1 = route1;
	}
	public Route getRoute2() {
		return route2;
	}
	public void setRoute2(Route route2) {
		this.route2 = route2;
	}
}
