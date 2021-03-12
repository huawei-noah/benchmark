package com.my.vrp.utils;

import java.util.ArrayList;
import java.util.Collections;

import com.my.vrp.Box;
/**
 * top序列类
 * @author dell
 *
 */
public class TopSequence {

	/**
	 * 存放top序列的箱子
	 * 适用于DBLF算法
	 */
	public  ArrayList<Box> topSequence = new ArrayList<Box>();
	/**
	 * 为top序列排序
	 */
	public  void topSort() {
		for(int i=0;i<this.topSequence.size()-1;i++) {
			for(int j=0;j<this.topSequence.size()-i-1;j++) {
				if(this.topSequence.get(j).getYCoor()+this.topSequence.get(j).getHeight()>
				this.topSequence.get(j+1).getYCoor()+this.topSequence.get(j+1).getHeight()) {
					Collections.swap(this.topSequence, j, j+1);
				}
			}
		}
	}
}
