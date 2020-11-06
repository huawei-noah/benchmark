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

package feasibilityCheck.util;


import java.nio.file.Paths;

public class Config {
    public static final double BOUNDARY_ERROR = 1.5;
    public static final double OVERLAP_ERROR = 1.5;
    public static final double CONTACT_ERROR = 1.;
    public static final double SUPPORT_RATIO = 0.8;
    public static final String TRUCK_TEMPLATE_PATH = Paths.get("base", "base.html.template").toString();
}
