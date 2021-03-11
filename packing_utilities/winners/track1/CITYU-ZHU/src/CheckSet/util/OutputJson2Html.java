/*
 *    Copyright (c) 2020. Huawei Technologies Co., Ltd.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
 */

package CheckSet.util;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import static CheckSet.util.DeleteDirectory.deleteDirectory;


public class OutputJson2Html {

    public static void Json2Html(
            String messagePath,
            String outputPath,
            String truckTemplatePath,
            String htmlPath
    ) throws IOException {
        String messageStr = new String(Files.readAllBytes(Paths.get(messagePath)));
        String outputStr = new String(Files.readAllBytes(Paths.get(outputPath)));
        toHtmlFiles(messageStr, outputStr, truckTemplatePath, htmlPath);
    }

    /**
     * Convert packing result to HTML files.
     */
    public static void toHtmlFiles(
            String messageStr,
            String estimateResult,
            String truckTemplatePath,
            String htmlPath
    ) throws IOException {
        // Generate HTML string.
        List<List<Map<String, Object>>> htmlStr = toHtmlStr(messageStr, estimateResult, truckTemplatePath);
        if (htmlStr.size() == 0) return;

        File dir = new File(htmlPath);
        if (dir.isDirectory()) {
            deleteDirectory(htmlPath);
        }
        dir.mkdir();

        int solutionIndex = 1;
        for (List<Map<String, Object>> solutionHtmlStrs: htmlStr) {
            String subDirPath = Paths.get(htmlPath, "solution" + solutionIndex).toString();
            File subDir = new File(subDirPath);
            subDir.mkdir();
            int truckIndex = 1;
            for (Map<String, Object> map: solutionHtmlStrs) {
                String truckHtmlStr = (String) map.get("truck");
                String truckTypeCode = (String) map.get("truckTypeCode");
                // Output truck-related html files.

                String truckFilename = Paths.get(subDirPath, truckIndex + "_" + truckTypeCode + ".html").toString();
                File truckFile = new File(truckFilename);
                truckFile.createNewFile();
                FileWriter truckWriter = new FileWriter(truckFile);
                truckWriter.write(truckHtmlStr);
                truckWriter.flush();
                truckWriter.close();
                truckIndex++;
            }
            solutionIndex++;
        }
    }

    /**
     * Convert input params and output params to HTML String.
     */
    public static List<List<Map<String, Object>>> toHtmlStr(
            String messageStr,
            String estimateResult,
            String truckTemplatePath
    ) throws IOException {
        List<List<Map<String, Object>>> result = new ArrayList<>();
        JSONObject packedResult = JSON.parseObject(estimateResult);
        JSONObject message = JSON.parseObject(messageStr);
        JSONObject algorithmBaseParamDto = message.getJSONObject(
                "algorithmBaseParamDto");
        JSONArray solutionArray = packedResult.getJSONArray("solutionArray");
        for (int i = 0; i < solutionArray.size(); i++) {
            JSONArray truckArray = (JSONArray) solutionArray.get(i);
            List<Map<String, Object>> solution = new ArrayList<>();
            for (int j = 0; j < truckArray.size(); j++) {
                JSONObject truck = (JSONObject) truckArray.get(j);
                JSONArray spuArray = truck.getJSONArray("spuArray");
                // Generate truck-related html strings.
                String boxesJsonStr = genBoxesJsonStr(spuArray);
                String truckJsonStr = genTruckJsonStr(truck, algorithmBaseParamDto.getJSONObject("truckTypeMap"));
                String truckHtmlStr = genTruckHtml(truckTemplatePath, boxesJsonStr, truckJsonStr);
                Map<String, Object> map = new HashMap<>();
                map.put("truckTypeCode", truck.getString("truckTypeCode"));
                map.put("truck", truckHtmlStr);
                solution.add(map);
            }
            result.add(solution);
        }
        return result;
    }

    /**
     * Generate JSON strings for a group of boxes.
     */
    private static String genBoxesJsonStr(JSONArray boxes) {
        List<String> boxJsonStrList = new ArrayList<>();
        for (int i = 0; i < boxes.size(); i++) {
            JSONObject box = (JSONObject) boxes.get(i);
            boxJsonStrList.add(genBoxJsonStr(box));
        }
        return String.format("[%s]", String.join(",", boxJsonStrList));
    }

    /**
     * Generate a JSON string for a single box.
     */
    private static String genBoxJsonStr(JSONObject box) {
        String material = "";
        int type = 1;

        return String.format(
                "{"
                        + "\"boxId\":\"%s\","
                        + "\"x\":%f,"
                        + "\"y\":%f,"
                        + "\"z\":%f,"
                        + "\"width\":%f,"
                        + "\"height\":%f,"
                        + "\"length\":%f,"
                        + "\"weight\":%f,"
                        + "\"platform\":\"%s\","
                        + "\"order\":%d,"
                        + "\"material\":\"%s\","
                        + "\"type\":%d,"
                        + "\"place\":%d"
                        + "}",
                box.getString("spuId"),
                box.getFloat("x") / 10f,
                box.getFloat("y") / 10f,
                box.getFloat("z") / 10f,
                box.getFloat("width") / 10f,
                box.getFloat("height") / 10f,
                box.getFloat("length") / 10f,
                box.getFloat("weight"),
                box.getString("platformCode"),
                box.getIntValue("order"),
                material,
                type,
                box.getIntValue("direction")
        );
    }

    /**
     * Generate JSON strings containing truck-related information.
     */
    private static String genTruckJsonStr(JSONObject truck, JSONObject truckTypeMap) {
        String truckTypeName = truckTypeMap.getJSONObject(truck.getString("truckTypeId")).getString(
                "truckTypeName");
        return String.format(
                "{"
                        + "\"container_length\":%f,"
                        + "\"container_width\":%f,"
                        + "\"container_height\":%f,"
                        + "\"container_max_load\":%f,"
                        + "\"container_type_name\":\"%s\""
                        + "}",
                truck.getFloat("innerLength") / 10,
                truck.getFloat("innerWidth") / 10,
                truck.getFloat("innerHeight") / 10,
                truck.getFloat("maxLoad"),
                truckTypeName
        );
    }

    /**
     * Generate truck-related HTML strings.
     */
    private static String genTruckHtml(
            String templatePath,
            String boxesJsonStr,
            String truckJsonStr
    ) throws IOException {
        String HTMLStr = new String(Files.readAllBytes(Paths.get(templatePath)));
        HTMLStr = HTMLStr.replaceAll("BOX_REPLACE_CONTENT", boxesJsonStr)
                .replaceAll("TRUCK_REPLACE_CONTENT", truckJsonStr);

        return HTMLStr;
    }

    public static void main(String[] args) {
        String messagePath = ".\\data\\data0923\\input\\";
        String outputPath = ".\\data\\out_params\\";
        String htmlPath = ".\\data\\visualization\\";
        File orders = new File(messagePath);
        for (File order: Objects.requireNonNull(orders.listFiles())) {
            String orderName = order.getName();
            if (orderName.endsWith("_d")) {
                orderName = orderName.replace("_d", "");
            }
            String singleMessagePath = order.getPath();
            String singleOutputPath = Paths.get(outputPath, orderName).toString();
            String singleHtmlPath = Paths.get(htmlPath, orderName).toString();
            try {
                Json2Html(
                        singleMessagePath,
                        singleOutputPath,
                        Config.TRUCK_TEMPLATE_PATH,
                        singleHtmlPath
                );
            }
            catch (Exception e) {
                System.out.println("Order " + orderName + " goes wrong: ");
                e.printStackTrace();
            }
        }
    }
}
