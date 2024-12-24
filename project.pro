# Nome do projeto
TEMPLATE = app
TARGET = main
CONFIG += console c++11
CONFIG -= app_bundle

# Diretórios de inclusão
INCLUDEPATH += /usr/local/include/opencv4
INCLUDEPATH += /usr/include/opencv4

# Diretórios de bibliotecas
LIBS += -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc

# Arquivos de origem
SOURCES += main.cpp

# Arquivos de cabeçalho

# Definir o sistema operacional, se necessário
unix:!macx {
    LIBS += -lpthread
}

# **Adicionar as flags para OpenMP**
QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp

# Qt Módulos necessários
QT += core gui widgets
