// threadsafe_queue.h
//
// Author: J.A. de Jong 
//
// Description:
// Implementation of a thread-safe queue, based on STL queue
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef THREADSAFE_QUEUE_H
#define THREADSAFE_QUEUE_H
#include <queue>
#include <mutex>
#include <condition_variable>

// A threadsafe-queue.
template <class T>
class SafeQueue {
  std::queue<T> _queue;
  mutable std::mutex _mutex;
  std::condition_variable _cv;
public:
    SafeQueue(): _queue(), _mutex() , _cv()
  {}

  ~SafeQueue(){}

  void enqueue(T t) {
    std::lock_guard<std::mutex> lock(_mutex);
    _queue.push(t);
    _cv.notify_one();
  }

  T dequeue() {
    std::unique_lock<std::mutex> lock(_mutex);
    while(_queue.empty())
    {
      // release lock as long as the wait and reaquire it afterwards.
      _cv.wait(lock);
    }
    T val = _queue.front();
    _queue.pop();
    return val;
  }
    bool empty() const {
        std::unique_lock<std::mutex> lock(_mutex);
        return _queue.size()==0;
    }
    size_t size() const {
        std::unique_lock<std::mutex> lock(_mutex);
        return _queue.size();
    }
};


#endif // THREADSAFE_QUEUE_H
//////////////////////////////////////////////////////////////////////
