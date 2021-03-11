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

package CheckSet.entity;

import CheckSet.util.ParseJsonList;

import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class Route {
    /**
     * Get the map of all platforms.
     */
    private Map allPlatformMap = new HashMap();

    @SuppressWarnings("unchecked")
    public Route(Map messageMap) {
        Map algorithmBaseParamDto = (Map) messageMap.get("algorithmBaseParamDto");
        List<Map> platformDtoList = ParseJsonList.parseJsonList(algorithmBaseParamDto.get("platformDtoList"), Map.class);
        for (Map platformDto: platformDtoList) {
            String platformCode = platformDto.get("platformCode").toString();
            allPlatformMap.put(platformCode, platformDto);
        }
    }

    public Map getAllPlatformMap() {
        return allPlatformMap;
    }
}
