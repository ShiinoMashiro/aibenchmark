/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MINDSPORE_INCLUDE_API_CALLBACK_TIME_MONITOR_H
#define MINDSPORE_INCLUDE_API_CALLBACK_TIME_MONITOR_H

#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include "mindspore/callback/callback.h"

namespace mindspore {

class TimeMonitor: public TrainCallBack {
 public:
  virtual ~TimeMonitor() = default;
  void EpochBegin(const TrainCallBackData &cb_data) override;
  CallbackRetValue EpochEnd(const TrainCallBackData &cb_data) override;
};
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_CALLBACK_TIME_MONITOR_H
