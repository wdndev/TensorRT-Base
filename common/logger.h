/**
 * @file        - logging.h
 * @author      - NVIDIA, wdn
 * @brief       - 日志
 * 
 * @copyright Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * 
 */
#ifndef TRT_LOGGING_H
#define TRT_LOGGING_H

#include <cassert>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

#include <NvInferRuntimeCommon.h>

using Severity = nvinfer1::ILogger::Severity;

/**
 * 日志缓存类
*/
class LogStreamConsumerBuffer : public std::stringbuf
{
public:
    LogStreamConsumerBuffer(std::ostream& stream, const std::string& prefix, bool should_log)
        : m_output(stream),
          m_prefix(prefix),
          m_should_log(should_log)
    {
    }

    LogStreamConsumerBuffer(LogStreamConsumerBuffer&& other)
        : m_output(other.m_output)
    {
    }

    ~LogStreamConsumerBuffer()
    {
        // std::streambuf::pbase() 给出了输出序列的缓冲部分的开头的指针
        // std::streambuf::pptr() 给出了输出序列当前位置的指针
        // 如果指向开头的指针不等于指向当前位置的指针，请调用 put_output() 将输出记录到流
        if(pbase() != pptr())
        {
            put_output();
        }
    }

    /**
     * 同步流缓冲区包括将缓冲区内容插入流中，重置缓冲区并刷新流
     * 
     * @return @c int   - 同步流缓冲区并取得成功返回0
    */
    virtual int sync()
    {
        put_output();
        return 0;
    }

    /**
     * 输出
    */
    void put_output()
    {
        if(m_should_log)
        {
            std::time_t time_stamp = std::time(nullptr);
            tm* tm_local = std::localtime(&time_stamp);
            std::cout << "[";
            std::cout << std::setw(4) << std::setfill('0') << 1900 + tm_local->tm_year << "-";
            std::cout << std::setw(2) << std::setfill('0') << 1 + tm_local->tm_mon << "-";
            std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_mday << " ";
            std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_hour << ":";
            std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_min << ":";
            std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_sec << "]";
            
            // 输出信息
            m_output << m_prefix << " : "<< str();
            // buffer置为空
            str("");
            // 清除stream
            m_output.flush();
        }
    }

    void set_should_log(bool should_log)
    {
        m_should_log = should_log;
    }


private:
    std::ostream& m_output;
    std::string m_prefix;
    bool m_should_log;
};

/**
 * 日志基类
*/
class LogStreamConsumerBase
{
public:
    LogStreamConsumerBase(std::ostream& stream, const std::string& prefix, bool should_log)
        : m_buffer(stream, prefix, should_log)
    {
    }
protected:
    LogStreamConsumerBuffer m_buffer;
};

/**
 * 日志类
*/
class LogStreamConsumer : protected LogStreamConsumerBase, public std::ostream
{
public:
    LogStreamConsumer(Severity reportable_severity, Severity severity)
        : LogStreamConsumerBase(severity_ostream(severity), severity_prefix(severity), severity <= reportable_severity),
        std::ostream(&m_buffer),    // 链接stream
        m_should_log(severity <= reportable_severity),
        m_severity(severity)
    {
    }

    LogStreamConsumer(LogStreamConsumer&& other)
        : LogStreamConsumerBase(severity_ostream(other.m_severity), severity_prefix(other.m_severity), other.m_should_log),
        std::ostream(&m_buffer),    // 链接stream
        m_should_log(other.m_should_log),
        m_severity(other.m_severity)
    {
    }

    void set_reportable_severity(Severity reportable_severity)
    {
        m_should_log = m_severity <= reportable_severity;
        m_buffer.set_should_log(m_should_log);
    }

private:
    static std::ostream& severity_ostream(Severity severity)
    {
        return severity > Severity::kINFO ? std::cout : std::cerr;
    }

    static std::string severity_prefix(Severity severity)
    {
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            return "[ IERROR]";
        case Severity::kERROR:
            return "[ ERROR ]";
        case Severity::kWARNING:
            return "[  WARN ]";
        case Severity::kINFO:
            return "[  INFO ]";
        case Severity::kVERBOSE:
            return "[ DEBUG ]";
        default:
            assert(0);
            return "[       ]";
        }
    }

    bool m_should_log;
    Severity m_severity;
};

/**
 * sample日志类
*/
class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kWARNING)
        : m_reportable_severity(severity)
    {
    }

    /**
     * 用于检索与此记录器相关联的nvinfer1::iLogger的前向兼容方法
    */
    nvinfer1::ILogger& get_trt_logger()
    {
        return *this;
    }

    /**
     * 实现输出
    */
    void log(Severity severity, const char* msg) override
    {
        LogStreamConsumer(m_reportable_severity, severity) << std::string(msg) << std::endl;
        //std::cout << std::string(msg) << std::endl;
    }

    /**
     * 设置控制日志输出的方法
    */
    void set_reportable_severity(Severity severity)
    {
        m_reportable_severity = severity;
    }

    /**
     * 获取控制日志输出的方法
    */
    Severity get_reportable_severity() const
    {
        return m_reportable_severity;
    }

private:
    /**
     * 返回消息前缀
    */
    static const char* severity_prefix(Severity severity)
    {
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            return "[ IERROR]";
        case Severity::kERROR:
            return "[ ERROR ]";
        case Severity::kWARNING:
            return "[  WARN ]";
        case Severity::kINFO:
            return "[  INFO ]";
        case Severity::kVERBOSE:
            return "[ DEBUG ]";
        default:
            assert(0);
            return "[       ]";
        }
    }
    
    /**
     * 消息流
    */
    static std::ostream& severity_ostream(Severity severity)
    {
        return severity > Severity::kINFO ? std::cout : std::cerr;
    }

    /**
     * 测试结构流
    */


    
    Severity m_reportable_severity;
};

namespace
{
/**
 * DEBUG 日志类
 * 
 * Example usage:
 * 
 *      LOG_DEBUG(logger) << "hello world" << std::endl;
*/
inline LogStreamConsumer LOG_DEBUG(const Logger& logger)
{
    return LogStreamConsumer(logger.get_reportable_severity(), Severity::kVERBOSE);
}

/**
 * INFO 日志类
 * 
 * Example usage:
 * 
 *      LOG_INFO(logger) << "hello world" << std::endl;
*/
inline LogStreamConsumer LOG_INFO(const Logger& logger)
{
    return LogStreamConsumer(logger.get_reportable_severity(), Severity::kINFO);
}

/**
 * WARN 日志类
 * 
 * Example usage:
 * 
 *      LOG_WARN(logger) << "hello world" << std::endl;
*/
inline LogStreamConsumer LOG_WARN(const Logger& logger)
{
    return LogStreamConsumer(logger.get_reportable_severity(), Severity::kWARNING);
}

/**
 * ERROR 日志类
 * 
 * Example usage:
 * 
 *      LOG_ERROR(logger) << "hello world" << std::endl;
*/
inline LogStreamConsumer LOG_ERROR(const Logger& logger)
{
    return LogStreamConsumer(logger.get_reportable_severity(), Severity::kERROR);
}

/**
 * FATAL 日志类
 * 
 * Example usage:
 * 
 *      LOG_FATAL(logger) << "hello world" << std::endl;
*/
inline LogStreamConsumer LOG_FATAL(const Logger& logger)
{
    return LogStreamConsumer(logger.get_reportable_severity(), Severity::kINTERNAL_ERROR);
}

} // namespace

#endif // TRT_LOGGING_H