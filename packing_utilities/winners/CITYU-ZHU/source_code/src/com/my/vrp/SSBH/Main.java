package com.my.vrp.SSBH;

import com.my.vrp.*;

import java.io.File;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Map;


import static com.my.vrp.Main.outputJSON;
import static com.my.vrp.SSBH.DataRead.readJSON;
import static com.my.vrp.SSBH.DataRead.setExtreme;


public class Main {
    public static void main(String[] args){
        /**
         * 初始化ideal和nadir points.
         * 这个map的格式为：文件名：{Optf1, Optf2, reff1, reff2}
         */
        Map<String, Double[]> idealNadirMap; //problemName->ideal&nadir
        String extremePath = "./data/extremes";
        idealNadirMap = setExtreme(extremePath);

        String inputPath = "./data/inputs_t";
        File f = new File(inputPath);
        String[] filenames = f.list();

        for(int file_idx=0;file_idx<filenames.length;file_idx++)
        {
            long begintime = System.nanoTime(); //开始计时
            process(inputPath, filenames[file_idx], idealNadirMap.get(filenames[file_idx]));
        }
    }

    public static void process(String basePath, String currentFileName, Double[] idealNadir){
        String fullFileName = basePath + "/" + currentFileName;
        Problem CurrentProblem;
        CurrentProblem = readJSON(fullFileName);
        ArrayList<Route> routes = new ArrayList<>();
        int truck_idx = 2;
//        for(int truck_idx=0;truck_idx<CurrentProblem.TRUCKTYPE_NUM; truck_idx++){
            for(int client_idx=0; client_idx<CurrentProblem.clients.size();client_idx++){
                ArrayList<Box> box_sequence = new ArrayList<Box>();
                box_sequence = CurrentProblem.clients.get(client_idx).getGoods();
                Route new_r = new Route((int)(1000+Math.random()*8999));
                LinkedList<Node> nodes = new LinkedList<Node>();
                nodes.add(new Node(CurrentProblem.depot_start));
                nodes.add(new Node(CurrentProblem.clients.get(client_idx)));
                nodes.add(new Node(CurrentProblem.depot_start));
                new_r.setNodes(nodes);
                Carriage want_truck = new Carriage(CurrentProblem.BASIC_TRUCKS.get(truck_idx));
                want_truck.setLength(want_truck.getLength() * 10.0);
                new_r.setCarriage(want_truck);
                new_r.ssbh_pack_long(box_sequence);
                new_r.getCarriage().setLength(new_r.start_forc);
                routes.add(new_r);
            }
//        }
        routes = Utils.ConcatRoute(routes, CurrentProblem);

        SolutionSet_vrp solutions = new SolutionSet_vrp();
        Solution_vrp testS = new Solution_vrp();
        testS.setRoutes(routes);
        solutions.add(testS);
//        outputJSON(solutions, fullFileName, CurrentProblem.PlatformIDCodeMap, "./data/outputs_t/E1597373872717");
//        outputJSON(testS, fullFileName, CurrentProblem.PlatformIDCodeMap, "./data/outputs/E1597195301689");

    }
}
