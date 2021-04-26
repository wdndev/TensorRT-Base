/**
 * @file        - timer.h
 * @author      - wdn (dongnianwang@outlook.com)
 * @brief       - CPU/GPU计时类
 * 
 * @copyright Copyright (c) 2021, wdn. All rights reserved.
 * 
 */
#ifndef TRT_TIME_H
#define TRT_TIME_H

#include <iostream>
#include <chrono>

#include <NvInferRuntimeCommon.h>
#include <cuda_runtime_api.h>

/**
 * Timer基类
*/
class TimerBase
{
public:
    /**
     * 计时开始
    */
    virtual void start() = 0;

    /**
     * 计时结束
    */
    virtual void stop() = 0;

    virtual void continuation() = 0;

    /**
     * 返回微妙
    */
    float microseconds() const noexcept
    {
        return m_ms * 1000.f;
    }

    /**
     * 返回毫秒
    */
    float milliseconds() const noexcept
    {
        return m_ms;
    }

    /**
     * 返回秒
    */
    float seconds() const noexcept
    {
        return m_ms / 1000.f;
    }

    /**
     * 重置
    */
    void reset() noexcept
    {
        m_ms = 0.f;
    }

protected:
    float m_ms{0.0f};
};

/**
 * 获取GPU时间
*/
template<typename Clock>
class CpuTimer : public TimerBase
{
public:
    using clock_type = Clock;

    /**
     * 开始
    */
    void start() override
    {
        reset();
        m_start = Clock::now();
    }

    /**
     * 继续
    */
    void continuation() override
    {
        m_start = Clock::now();
    }

    /**
     * 停止
    */
    void stop() override
    {
        m_stop = Clock::now();
        m_ms += std::chrono::duration<float, std::milli> {m_stop - m_start}.count();
    }

private:
    std::chrono::time_point<Clock> m_start;
    std::chrono::time_point<Clock> m_stop;
};

/**
 * GPU计时
*/
class GpuTimer : public TimerBase
{
public:
    GpuTimer(cudaStream_t stream) : m_stream(stream)
    {
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(m_start);
        cudaEventDestroy(m_stop);
    }

    /**
     * 开始
    */
    void start() override
    {
        reset();
        cudaEventRecord(m_start, m_stream);
    }

    /**
     * 继续
    */
    void continuation() override
    {
        cudaEventRecord(m_start, m_stream);
    }

    /**
     * 停止
    */
    void stop() override
    {
        cudaEventRecord(m_stop, m_stream);
        float ms{0.0f};
        cudaEventSynchronize(m_stop);
        cudaEventElapsedTime(&ms, m_start, m_stop);
        m_ms += ms;
    }

private:
    cudaEvent_t m_start;
    cudaEvent_t m_stop;
    cudaStream_t m_stream;
};

#endif // TRT_TIME_H