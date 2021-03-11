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

import java.util.ArrayList;
import java.util.List;

public class ParseJsonList {
    public static <T> List<T> parseJsonList(Object jsonObj, Class<T> clazz) {
        List<T> list = new ArrayList<>();
        if (jsonObj == null) {
            return list;
        }
        return JSON.parseArray(jsonObj.toString(), clazz);
    }
}
