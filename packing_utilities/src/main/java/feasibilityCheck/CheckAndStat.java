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

package feasibilityCheck;

import java.io.IOException;

public class CheckAndStat {
    public static void main(String[] args) throws IOException {
        String inputDir = ".\\data\\data0923\\input";
        String outputDirOld = ".\\data\\result1";
        String outputDirNew = ".\\data\\result3";
        String checkResultFile = ".\\result\\checkResult3.txt";
        String statResultFile = ".\\result\\statResult3.txt";
        MultiCheck multiCheck = new MultiCheck(inputDir, outputDirNew, checkResultFile);
        multiCheck.check();
        Statistics statistics = new Statistics(inputDir, outputDirOld, outputDirNew, statResultFile);
        statistics.calc();
    }
}
