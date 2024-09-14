/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_INCLUDE_API_GRAPH_H
#define MINDSPORE_INCLUDE_API_GRAPH_H

#include <cstddef>
#include <vector>
#include <map>
#include <memory>
#include "mindspore/status.h"
#include "mindspore/types.h"

namespace mindspore {
class NetData;
class Net;

class MS_API Graph {
 public:
  class GraphData;
  enum Type : uint32_t {
    kExpressionGraph = 0,  ///< graph as expression - can auto grad
    kExecutableGraph = 1,  ///< graph is loaded as is
    kUnknownTypeGraph = 0xffffffff
  };
  Graph();
  explicit Graph(const std::shared_ptr<GraphData> &graph_data);
  explicit Graph(std::shared_ptr<GraphData> &&graph_data);
  explicit Graph(std::nullptr_t);
  ~Graph();
  explicit Graph(Type executable);
  explicit Graph(Net *net);

  enum ModelType ModelType() const;
  bool operator==(std::nullptr_t) const;
  bool operator!=(std::nullptr_t) const;
  bool IsExecutable() { return graph_type_ == kExecutableGraph; }

 private:
  friend class GraphCell;
  friend class ModelImpl;
  friend class NetImpl;
  friend class Model;
  std::shared_ptr<GraphData> graph_data_;
  std::shared_ptr<NetData> net_data_;
  Type graph_type_ = kExecutableGraph;
};
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_GRAPH_H
