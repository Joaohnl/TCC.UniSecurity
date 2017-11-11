package processing;

/* 
    Este painel repetidamente tira imagens do dispositivo de captura e as desenha no painel.
    Uma face Ã© destacada com um retÃ¢ngulo verde, que Ã© atualizado conforme a face se movimenta.
    
    A face destacada pode ser armazenada para reconhecimento da pessoa.

    A tarefa de detecÃ§Ã£o Ã© realizado pelo Haar cascade classifier disponibilizado pelo JavaCV.
    Esta tarefa Ã© executada em sua prÃ³pria thread devido o processo ser mais lento. sendo assim
    a captura das imagens nÃ£o Ã© afetada pelo reconhecimento dos rostos.
 */
import gui.TelaLog;
import java.awt.*;
import javax.swing.*;
import java.awt.image.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;
import static org.bytedeco.javacpp.helper.opencv_objdetect.cvHaarDetectObjects;
import org.bytedeco.javacpp.opencv_objdetect.CvHaarClassifierCascade;
import org.bytedeco.javacpp.videoInputLib.videoInput;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter.ToIplImage;

public class PainelReconhecimento extends JPanel implements Runnable {

    private TelaLog log;

    /* DimensÃµes de cada imagem; o painel Ã© do mesmo tamanho das imagens */
    private static final int WIDTH = 640;
    private static final int HEIGHT = 480;

    private static final int DELAY = 100;  // Tempo (ms) delay para desenhar o painel

    private static final int CAMERA_ID = 0;

    private static final int IM_SCALE = 4;
    private static final int SMALL_MOVE = 5;

    // tempo (ms) entre cada detecÃ§Ã£o de face
    private static final int DETECT_DELAY = 500;

    private static final int MAX_TASKS = 4;
    // Nr. mÃ¡ximo de tarefas que podem ficar aguardando no executor

    // Modelo cascata utilizado para detecÃ§Ã£o das faces (Disponibilizado pelo OpenCV)
    private static final String FACE_CASCADE_FNM = "haarcascade_frontalface_alt.xml";
    // "haarcascade_frontalface_alt2.xml";

    // Atributos para salvar uma imagem
    private static final String FACE_DIR = "savedFaces";
    private static final String FACE_FNM = "face";
    private static String FACE_ID;
    private static final int FACE_WIDTH = 125;
    private static final int FACE_HEIGHT = 150;

    private IplImage snapIm = null;
    private volatile boolean isRunning;
    private volatile boolean isFinished;

    // Usado para disponibilizar tempo da captura das imagens
    private int imageCount = 0;
    private long totalTime = 0;
    private Font msgFont;

    // Variáveis do JavaCV
    private FrameGrabber grabber = null;
    private CvHaarClassifierCascade classifier;
    private CvMemStorage storage;
    private CanvasFrame debugCanvas;
    private IplImage grayIm;

    // Usado para as threads que executam a detecÃ§Ã£o das faces
    private ExecutorService executor;
    private AtomicInteger numTasks;

    // Usado para armazenar nÃºmero de tarefas de detecÃ§Ã£o
    private long detectStartTime = 0;

    private String identificacao;           // IdentificaÃ§Ã£o da pessoa pelo LBPH 
    private IplImage[] facesTreinamento;    // Armazena as faces para treinamento do LBPH
    private LBPHFaceRecognizer lbphFaceRecognizer;
    private Rectangle faceRect;             // Armazena as coordenadas da face

    private volatile boolean saveFace = false;

    public PainelReconhecimento() {
        setBackground(Color.white);
        msgFont = new Font("SansSerif", Font.BOLD, 18);

        executor = Executors.newSingleThreadExecutor();
        /* O executor controla uma Ãºnica thread com uma fila.
            Somente uma tarefa pode executar de cada vez. As outras devem esperar.
         */
        numTasks = new AtomicInteger(0);
        // Usado para limitar o tamanho da fila do executor.

        initDetector();
        faceRect = new Rectangle();
        lbphFaceRecognizer = new LBPHFaceRecognizer();
        facesTreinamento = new IplImage[lbphFaceRecognizer.getNUM_IMAGES_PER_PERSON()];

        isRunning = true;
        isFinished = false;

        log = new TelaLog();
        log.setVisible(true);

        new Thread(this).start();   // Atualiza as imagens para o painel.
    }

    private void initDetector() {
        // Cria uma instÃ¢ncia do cascade classifier para detecÃ§Ã£o dos rostos
        classifier = new CvHaarClassifierCascade(cvLoad(FACE_CASCADE_FNM));
        if (classifier.isNull()) {
            System.out.println("\nCould not load the classifier file: " + FACE_CASCADE_FNM);
            System.exit(1);
        }

        storage = CvMemStorage.create();  //Cria armazenamento usado para detecÃ§Ã£o

        // debugCanvas = new CanvasFrame("Debugging Canvas");
        // useful for showing JavaCV IplImage objects, to check on image processing
    }  // end of initDetector()

    /*
        Deixa o painel no tamanho suficiente para uma imagem
     */
    public Dimension getPreferredSize() {
        return new Dimension(WIDTH, HEIGHT);
    }

    /*
        Exibe a imagem do dispositivo de captura a cada tempo de DELAY.
        A tarefa de detecÃ§Ã£o somente se inicia apÃ³s cada tempo de DETECT_DELAY
        e somente se o nÃºmero de tarefas no executor for menor que seu limite
     */
    public void run() {
        grabber = initGrabber(CAMERA_ID);
        if (grabber == null) {
            return;
        }

        long duration;

        while (isRunning) {
            long startTime = System.currentTimeMillis();

            snapIm = picGrab(grabber, CAMERA_ID);

            if (((System.currentTimeMillis() - detectStartTime) > DETECT_DELAY)
                    && (numTasks.get() < MAX_TASKS)) {
                trackFace(snapIm);
            }
            imageCount++;
            repaint();
            super.repaint();

            duration = System.currentTimeMillis() - startTime;
            totalTime += duration;

            if (duration < DELAY) {
                try {
                    Thread.sleep(DELAY - duration);  // Aguarda terminar o tempo de delay
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            }

        }
        closeGrabber(CAMERA_ID);
        log.setjTextArea1("Fim da execução");
        isFinished = true;

    }

    /*
        Inicia objeto de captura
     */
    private FrameGrabber initGrabber(int ID) {
        log.setjTextArea1("Inicializando captura pelo dispositivo: " + videoInput.getDeviceName(ID) + "\n");
        //System.out.println("Inicializando captura pelo dispositivo: " + videoInput.getDeviceName(ID));
        try {
            if (grabber == null) {
                grabber = FrameGrabber.createDefault(ID);
                grabber.setFormat("dshow");       // Usando DirectShow
                grabber.setImageWidth(WIDTH);     // tamanho padrÃ£o das imagens Ã© pequeno: 320x240
                grabber.setImageHeight(HEIGHT);
            }
            grabber.start();
        } catch (Exception e) {
            System.out.println("Não foi possível carregar a câmera!");
            System.out.println(e);
            System.exit(1);
        }
        return grabber;
    }

    private IplImage picGrab(FrameGrabber grabber, int ID) {
        IplImage im = null;
        OpenCVFrameConverter.ToIplImage conversor = new OpenCVFrameConverter.ToIplImage();
        try {
            im = conversor.convert(grabber.grab());  //Tira uma foto
        } catch (Exception e) {
            System.out.println("Problema ao carregar imagem da câmera " + ID);
        }
        return im;
    }

    /*
        Finaliza a obtenção das imagens do dispositivo de cÃ¢mera
     */
    public void closeGrabber(int ID) {
        try {
            grabber.stop();
            grabber.release();
        } catch (Exception e) {
            System.out.println("Problema ao desativar captura da câmera " + ID);
        }
    }

    /*
        Desenha a imagem, o retângulo em volta a da face detectada e a média de tempo
        de obtenção das imagens da cÃ¢mera no canto inferior esquerdo do painel.
        No canto superior exibe a face identificada.
        O tempo exibido não inclui a tarefa de detecçãoo do rosto.
     */
    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        g2.setFont(msgFont);

        // Desenha a imagem, estatÃ­sticas e retÃ¢ngulo
        if (snapIm != null) {
            g2.setColor(Color.YELLOW);
            OpenCVFrameConverter.ToIplImage grabberConverter = new OpenCVFrameConverter.ToIplImage();
            Java2DFrameConverter paintConverter = new Java2DFrameConverter();
            org.bytedeco.javacv.Frame frame = grabberConverter.convert(snapIm);
            g2.drawImage(paintConverter.getBufferedImage(frame, 1), 0, 0, this);   // Desenha a imagem
            String statsMsg = String.format("Tempo médio de captura:  %.1f ms",
                    ((double) totalTime / imageCount));

            // Escreve as estatÃ­sticas no canto inferior esquerdo
            g2.drawString(statsMsg, 5, HEIGHT - 10);

            drawRect(g2);
        } else {  // Caso ainda nÃ£o obtiver nenhuma imagem
            g2.setColor(Color.BLUE);
            g2.drawString("Carregando câmera: " + CAMERA_ID, 5, HEIGHT - 10);
        }
    }

    /*
        Usa o retÃ¢ngulo da face a desenhar o retÃ¢ngulo verde em torno da face
        O desenho do retÃ¢ngulo estÃ¡ em um bloco sincronizado pois a variÃ¡vel faceRect pode estar sendo
        utilizada em outra thread.
     */
    private void drawRect(Graphics2D g2) {
        synchronized (faceRect) {
            if (faceRect.width == 0) {
                return;
            }

            // Desenha o retÃ¢ngulo
            g2.setColor(Color.GREEN);
            g2.setStroke(new BasicStroke(2));
            g2.drawRect(faceRect.x, faceRect.y, faceRect.width, faceRect.height);

        }
    }

    /*  Termina a execuÃ§Ã£o do run() e espera a finalizaÃ§Ã£o do sistema.
        Para a aplicaÃ§Ã£o atÃ© que sua execuÃ§Ã£o termine.
     */
    public void closeDown() {
        isRunning = false;
        log.dispose();
        while (!isFinished) {
            try {
                Thread.sleep(DELAY);
            } catch (Exception ex) {
            }
        }
    }

    // ------------------------- face tracking ----------------------------\\
    /*  Cria um thread para detectar faces nas imagens geradas pela cÃ¢mera.
        Armazena as coordenadas das faces em faceRect e salva a imagem para treinamento caso solicitado.
        Imprime o tempo de execuÃ§Ã£o no console
     */
    private void trackFace(IplImage img) {
        numTasks.getAndIncrement();     // Incrementa o nÃºmero de tarefas antes de entrar na fila
        executor.execute(new Runnable() {
            public void run() {
                detectStartTime = System.currentTimeMillis();
                CvRect rect = findFace(img);
                if (rect != null) {

                    // Seta parÃ¢metros do retÃ¢ngulo
                    setRectangle(rect);
                } else {
                    faceRect.setBounds(0, 0, 0, 0);
                }
                long detectDuration = System.currentTimeMillis() - detectStartTime;
                log.setjTextArea1(" duração da detecção: " + detectDuration + "ms\n");
                //System.out.println(" detection duration: " + detectDuration + "ms");
                numTasks.getAndDecrement();  // Decrementa o nÃºmero de tarefas quando terminado
            }
        });
    }

    /*  Formata a imagem e converte para escala de cinza. FormataÃ§Ã£o deixa a imagem menor
        tornando o processamento mais rÃ¡pido.
        O detector Haar precisa como parÃ¢metro a imagem em escala de cinza    
     */
    private IplImage scaleGray(IplImage img) {
        // Converte para escalas de cinza
        IplImage grayImg = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
        cvCvtColor(img, grayImg, CV_BGR2GRAY);

        // Formata a imagem
        IplImage smallImg = IplImage.create(grayImg.width() / IM_SCALE,
                grayImg.height() / IM_SCALE, IPL_DEPTH_8U, 1);
        cvResize(grayImg, smallImg, CV_INTER_LINEAR);

        // Equaliza a imagem menor em escalas de cinza
        cvEqualizeHist(smallImg, smallImg);
        return smallImg;
    }

    /*  Detecta somente uma face utilizando o Haar Detector
        Chama a funÃ§Ã£o para a identificaÃ§Ã£o do rosto encontrado.
     */
    private CvRect findFace(IplImage img) {

        // Converte para escalas de Cinza
        grayIm = scaleGray(img);

        /*
     // Mostra a imagem em escalas de cinza para verificar o processo de tratamento da imagem
     debugCanvas.showImage(grayIm);
	 debugCanvas.waitKey(0);
         */
        // System.out.println("Detecting largest face...");   // cvImage
        CvSeq faces = cvHaarDetectObjects(grayIm, classifier, storage, 1.1, 1, // 3
                // CV_HAAR_SCALE_IMAGE |
                CV_HAAR_DO_ROUGH_SEARCH | CV_HAAR_FIND_BIGGEST_OBJECT);
        // Por questÃµes de seguranÃ§a e rapidez, sÃ³ serÃ¡ detectado o rosto maior e mais prÃ³ximo.

        int total = faces.total();
        if (total == 0) {
            log.setjTextArea1("Face não encontrada\n");
            //System.out.println("No faces found");
            return null;
        } else if (total > 1) //Este caso não deveria ocorrer. IncluÃ­do por seguranÃ§a
        {
            log.setjTextArea1("Multiplas faces detectadas (" + total + "); utilizando a primeira\n");
            // System.out.println("Multiple faces detected (" + total + "); using the first");
        } else {
            log.setjTextArea1("Face detectada\n");
            //System.out.println("Face detected");
        }

        CvRect rect = new CvRect(cvGetSeqElem(faces, 0));

        if (saveFace) {
            learnNewFace(img);
        }

        //Identifica o rosto
        identificacao = lbphFaceRecognizer.identifyFace(grayIm);

        cvClearMemStorage(storage);
        return rect;
    }

    /* Extrai o tamanho e as coordenadas da imagem desatacada da estrutura do retÃ¢ngulo do JavaCV
       e armazena em um retÃ¢ngulo Java.
       Durante o processo, desfaz o escalonamento que foi aplicado a imagem antes da detecÃ§Ã£o do rosto.
       Disponibiliza informaÃ§Ãµes de movimento do rosto na imagem.
       O uso dessa funÃ§Ã£o estÃ¡ em um bloco sincronizado pois o retÃ¢ngulo pode estar sendo utilizado para
       atualizar o painel de imagem ou pintura do retÃ¢ngulo ao mesmo tempo em outras threads.
     */
    private void setRectangle(CvRect r) {
        synchronized (faceRect) {
            int xNew = r.x() * IM_SCALE;
            int yNew = r.y() * IM_SCALE;
            int widthNew = r.width() * IM_SCALE;
            int heightNew = r.height() * IM_SCALE;

            // calcula o movimento do retÃ¢ngulo comparado com o anterior
            int xMove = (xNew + widthNew / 2) - (faceRect.x + faceRect.width / 2);
            int yMove = (yNew + heightNew / 2) - (faceRect.y + faceRect.height / 2);

            // DispÃµe informaÃ§Ãµes de movimento da face se for significante
            if ((Math.abs(xMove) > SMALL_MOVE) || (Math.abs(yMove) > SMALL_MOVE)) {
                log.setjTextArea1("Movimento (x,y): (" + xMove + "," + yMove + ")\n");
                //System.out.println("Movement (x,y): (" + xMove + "," + yMove + ")");
            }

            faceRect.setRect(xNew, yNew, widthNew, heightNew);
        }
    }

    // --------------------------- Salvar e Aprender nova Face -----------------------------------
    public void saveFace(String faceID) {
        saveFace = true;
        FACE_ID = faceID;
    }

    /* Armazena e aprende nova Face */
    private void learnNewFace(IplImage img) {
        numTasks.getAndIncrement();         // Incrementa nÃºmero de tarefas antes de entrar na fila.
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {

                    facesTreinamento[0] = scaleGray(img);

                    lbphFaceRecognizer.saveNewFace(FACE_ID, facesTreinamento);
                    saveFace = false;
                } catch (Exception ex) {
                    Logger.getLogger(PainelReconhecimento.class.getName()).log(Level.SEVERE, null, ex);
                }
                numTasks.getAndDecrement();             //Decrementa nÃºmero de tarefas quando execuÃ§Ã£o termina
            }
        });
    }

    /* Corta a imagem utilizando do quadrado que possui as coordenadas da face
     O uso dessa funÃ§Ã£o estÃ¡ em um bloco sincronizado pois o retÃ¢ngulo pode estar sendo utilizado para
     atualizar o painel de imagem ou pintura do retÃ¢ngulo ao mesmo tempo em outras threads.
     
     */
    private IplImage clipSaveFace(IplImage img) {
        BufferedImage clipIm = null;

        synchronized (faceRect) {
            if (faceRect.width == 0) {
                System.out.println("Nenhuma face selecionada");
                return img;
            }
            OpenCVFrameConverter.ToIplImage grabberConverter = new OpenCVFrameConverter.ToIplImage();
            Java2DFrameConverter paintConverter = new Java2DFrameConverter();
            org.bytedeco.javacv.Frame frame = grabberConverter.convert(img);
            BufferedImage im = paintConverter.getBufferedImage(frame, 1);
            try {
                clipIm = im.getSubimage(faceRect.x, faceRect.y, faceRect.width, faceRect.height);
            } catch (RasterFormatException e) {
                System.out.println("Could not clip the image");
            }
        }
        if (clipIm != null) {
            BufferedImage grayIm = resizeImage(clipIm);
            BufferedImage faceIm = clipToFace(grayIm);
            return toIplImage(faceIm);
        } else {
            return img;
        }
    }

    /* Altera a imagem para um tamanho padrÃ£o e a transforma em escalas de cinza */
    private BufferedImage resizeImage(BufferedImage im) {
        int imWidth = im.getWidth();
        int imHeight = im.getHeight();
        System.out.println("Original (w,h): (" + imWidth + ", " + imHeight + ")");

        double widthScale = FACE_WIDTH / ((double) imWidth);
        double heightScale = FACE_HEIGHT / ((double) imHeight);
        double scale = (widthScale > heightScale) ? widthScale : heightScale;

        int nWidth = (int) Math.round(imWidth * scale);
        int nHeight = (int) Math.round(imHeight * scale);

        // Converte para escalas de cinza enquanto muda seu tamanho
        BufferedImage grayIm = new BufferedImage(nWidth, nHeight,
                BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2 = grayIm.createGraphics();
        g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2.drawImage(im, 0, 0, nWidth, nHeight, 0, 0, imWidth, imHeight, null);
        g2.dispose();

        System.out.println("Escalas de Cinza (w,h): (" + nWidth + ", " + nHeight + ")");
        return grayIm;
    }


    /* Corta a imagem em no tamanho FACE_WIDTHxFACE-HEIGHT
       Assume que a imagem do parÃ¢metro Ã© do tamanho da face ou maior
     */
    private BufferedImage clipToFace(BufferedImage im) {
        int xOffset = (im.getWidth() - FACE_WIDTH) / 2;
        int yOffset = (im.getHeight() - FACE_HEIGHT) / 2;
        BufferedImage faceIm = null;
        try {
            faceIm = im.getSubimage(xOffset, yOffset, FACE_WIDTH, FACE_HEIGHT);
            System.out.println("Imagem cortada para as dimensões ("
                    + FACE_WIDTH + ", " + FACE_HEIGHT + ")");
        } catch (RasterFormatException e) {
            System.out.println("Não foi possível cortar imagem");
            OpenCVFrameConverter.ToIplImage grabberConverter = new OpenCVFrameConverter.ToIplImage();
            Java2DFrameConverter paintConverter = new Java2DFrameConverter();
            org.bytedeco.javacv.Frame frame = grabberConverter.convert(grayIm);
            faceIm = paintConverter.getBufferedImage(frame);
        }
        return faceIm;
    }

    /* Converte uma BufferedImage para IplImage para utilizaÃ§Ã£o do OpenCV */
    private IplImage toIplImage(BufferedImage bufImage) {
        ToIplImage iplConverter = new OpenCVFrameConverter.ToIplImage();
        Java2DFrameConverter java2dConverter = new Java2DFrameConverter();
        IplImage iplImage = iplConverter.convert(java2dConverter.convert(bufImage));
        return iplImage;
    }

    public processing.LBPHFaceRecognizer getLbphFaceRecognizer() {
        return lbphFaceRecognizer;
    }

    public void setLbphFaceRecognizer(processing.LBPHFaceRecognizer lbphFaceRecognizer) {
        this.lbphFaceRecognizer = lbphFaceRecognizer;
    }

} // end of PainelReconhecimento class

