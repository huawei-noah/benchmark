#     Copyright (c) 2020. Huawei Technologies Co., Ltd.
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#

import os
import sys
from pack import Pack


def main(argv):
    input_dir = argv[1]
    output_dir = argv[2]
    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            message_str = f.read()
        pack = Pack(message_str,
                    output_path=output_dir)
        pack.run()


if __name__ == "__main__":
    main(sys.argv)
