package com.my.vrp.utils;

import java.util.ArrayList;
import java.util.Collections;

import com.my.vrp.Box;
/**
 * back序列类
 * @author dell
 *
 */
public class BackSequence {

	/**
	 * 存放back序列箱子
	 * 适用DBLF算法
	 */
	public  ArrayList<Box> backSequence = new ArrayList<Box>();
	/**
	 * 为back序列排序
	 */
	public  void backSort() {
		for(int i=0;i<this.backSequence.size()-1;i++) {
			for(int j=0;j<this.backSequence.size()-i-1;j++) {
				if(this.backSequence.get(j).getZCoor()+this.backSequence.get(j).getLength()>
				this.backSequence.get(j+1).getZCoor()+this.backSequence.get(j+1).getLength()) {
					Collections.swap(this.backSequence, j, j+1);
				}
			}
		}
	}
}
