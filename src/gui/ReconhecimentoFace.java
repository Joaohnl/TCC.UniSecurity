package gui;


/* Mostra uma sequência de imagens capturadas da webcam em um Painel (PainelReconhecimento).
   Uma face é destacada com um quadrado verde.
   É possível salvar a imagem destacada apertado o botão salvar face.

 */
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import processing.PainelReconhecimento;

public class ReconhecimentoFace extends JFrame {
    // Componentes da interface do usuário

    private PainelReconhecimento facePanel;

    public ReconhecimentoFace() {
        super("Reconhecimento Facial");

        Container c = getContentPane();
        c.setLayout(new BorderLayout());

        facePanel = new PainelReconhecimento(); // sequência de imagens aparece aqui.
        c.add(facePanel, BorderLayout.CENTER);

        // Botão para salvar face detectada
        JButton btnSalvarFace = new JButton("Salvar face");
        JButton btnReconhecimento = new JButton("Iniciar reconhecimento");


        JTextField txtIdentificacao = new JTextField("Insira o RA do aluno");

//        txtIdentificacao.addKeyListener(new KeyAdapter() {
//            // Habilita ou desabilita o botão de salvar face.
//            // Campo do RA precisa ter 9 digítos (tamanho do RA dos alunos) para habilitar o botão.
//            @Override
//            public void keyReleased(KeyEvent e) { // Observa entrada de dígitos no campo
//                try {
//                    int teste = Integer.parseInt(txtIdentificacao.getText());
//                    if (txtIdentificacao.getText().length() != 9) {
//                        btnSalvarFace.setEnabled(false);
//                    } else {
//                        btnSalvarFace.setEnabled(true);
//                    }
//                } catch (NumberFormatException exc) {
//                    txtIdentificacao.setText("");
//                }
//
//            }
//        });

        JPanel p = new JPanel();
        p.add(btnReconhecimento);
        p.add(txtIdentificacao);
        p.add(btnSalvarFace);
        c.add(p, BorderLayout.SOUTH);

        addWindowListener(new WindowAdapter() {
            public void windowClosed(WindowEvent e) {
                facePanel.closeDown();    // para de tirar imagens do dispositivo de câmera
                
            }
        });
        
        
        
        
        
        setDefaultCloseOperation(DISPOSE_ON_CLOSE);

        setResizable(false);
        pack();
        setLocationRelativeTo(null);
        setVisible(true);
    }
} //Fim da classe telaPrincipal
