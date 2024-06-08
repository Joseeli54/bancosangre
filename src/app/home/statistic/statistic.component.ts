import { Component, ElementRef, Input, OnInit, ViewChild } from '@angular/core';
import { Table } from 'primeng/table';
import { Diagnostico } from 'src/app/models/diagnostico';
import { Sangre } from 'src/app/models/sangre';
import { UserServices } from '../../services/userservices.services';
import { NgbModal, ModalDismissReasons } from '@ng-bootstrap/ng-bootstrap';
import { AlertService } from '../../services/alert/alert.service';
import { environment } from '../../../environments/environment';
import { User } from 'src/app/models/user';
import { Chart } from 'chart.js/auto';


@Component({
  selector: 'app-statistic',
  templateUrl: './statistic.component.html',
  styleUrls: ['./statistic.component.css']
})

export class StatisticComponent implements OnInit {

  public content_sangres : Array<Sangre> = [];
  public content_diagnosticos : Array<Diagnostico> = [];
  public content_usuarios : Array<User> = [];

  constructor(public userService: UserServices,
    private modalService: NgbModal,
    private alertService: AlertService) { 
    localStorage.setItem('view', '2');
  }

  ngOnInit(): void {
    this.getSangre();
  }

  public getCantidadSangres(){
    return this.content_sangres.length;
  }

  public getCantidadDiagnosticos(){
    return this.content_diagnosticos.length;
  }

  public getCantidadUsuarios(){
    return this.content_usuarios.length;
  }

  public getSangre(){
    localStorage.setItem('spinner', '1');

    this.userService
      .getUrl('sangres')
      .then(response => {
        this.content_sangres = response;
        console.log(this.content_sangres)
        localStorage.setItem('spinner', '0');

        this.getSangrePorGrupo();
        this.getDiagnosticos();
      })
      .catch(error => {
        console.log(error.error)
        this.alertService.errorOcurred(error.statusText);
      });
  }

  public getUsers(){
    this.userService.getUrl('users')
    .then(data => {

      this.content_usuarios = [];
      for (let index = 0; index < data.length; index++) {
        if(data[index].role == "USER"){
          this.content_usuarios.push(data[index]);
        }
      }

      this.getDonantesPorSangre();
    })
    .catch(err => {
      console.log(err);
    })
  }

  public getDiagnosticos(){
    localStorage.setItem('spinner', '1');

    this.userService
      .getUrl('diagnosticos')
      .then(response => {
        this.content_diagnosticos = response;
        console.log(this.content_diagnosticos)
        localStorage.setItem('spinner', '0');
        this.getDiagnosticoPorSangre();
        this.getUsers();
      })
      .catch(error => {
        console.log(error)
        this.alertService.errorOcurred(error.statusText);
      });
  }

  public categories = [
    { code: "1", title: 'A' },
    { code: "2", title: 'AB' },
    { code: "3", title: 'B' },
    { code: "4", title: 'O' },
  ];

  public categoryName(code : any){
    let name = "No Aplicado";

    for (let index = 0; index < this.categories.length; index++) {
      if(this.categories[index].code == code){
        name = this.categories[index].title;
      }
    }

    return name;
  }

  public dynamicColors(count : any){
    var arrayColors = [
      "#343a40","#8150bd","#007bff","#212529",
      "#28a745","#dc3545","#ffc107","#007bff",
      "#dc3545","#ff4500","#fd7e14","#17a2b8"
    ];

    var colores;
    var bCount = 0;

    if(count > 1){
      while(bCount == 0){
        colores = [];
        for (let index = 0; index < count; index++) {
          colores.push(arrayColors[Math.floor(Math.random()*arrayColors.length)])
        }

        colores = colores.filter((item, index, self) => self.indexOf(item) === index);

        if(colores.length == count){
          bCount = 1;
        }
      }

      return colores;
    }

    return arrayColors[Math.floor(Math.random()*arrayColors.length)];
  }

  public getSangrePorGrupo(){
    const ctx = document.getElementById('myChart1') as HTMLCanvasElement;
    var labelGroups = [];
    var valueGroups = [];
    var backgroundGroup : any;

    for (let index = 0; index < this.categories.length; index++) {
      for (let index2 = 0; index2 < this.content_sangres.length; index2++) {
        if(this.content_sangres[index2].grupo == this.categories[index].code){
          labelGroups.push(this.categories[index].title);
        }
      }
    }

    labelGroups = labelGroups.filter((item, index, self) => self.indexOf(item) === index);

    for (let index = 0; index < labelGroups.length; index++) {
      valueGroups.push(0);
      for (let index2 = 0; index2 < this.content_sangres.length; index2++) {
        if(this.categoryName(this.content_sangres[index2].grupo) == labelGroups[index]){
          valueGroups[index] = valueGroups[index] + 1;
        }
      }
    }

    backgroundGroup = this.dynamicColors(labelGroups.length);

    const myChart = new Chart(ctx, {
      type: 'doughnut', // Tipo de gráfico (puedes elegir otros como 'line', 'pie', etc.)
      data: {
        labels: labelGroups, // Etiquetas para el eje X
        datasets: [{
          label: 'Donantes',
          data: valueGroups, // Datos para el eje Y
          backgroundColor: backgroundGroup , // Color de fondo de las barras
          borderColor: backgroundGroup, // Color del borde de las barras
          borderWidth: 1, // Ancho del borde
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            position: 'right',
            display: true,
          },
          title: {
            display: false,
          }
        },
        maintainAspectRatio: false,
        elements: {
          point: {
            radius: 1
          }
        }
      }
    });
  }

  public sangreName(id : any){
    let name = "No Aplicado";

    for (let index = 0; index < this.content_sangres.length; index++) {
      if(this.content_sangres[index]._id == id){
        name = this.content_sangres[index].name!;
      }
    }

    return name;
  }

  public sangreGrupoByIdUser(id : any){
    let name = "No Aplicado";

    for (let index = 0; index < this.content_sangres.length; index++) {
      if(this.content_sangres[index].id_user == id){
        name = this.categoryName(this.content_sangres[index].grupo!);
      }
    }

    return name;
  }

  public getDiagnosticoPorSangre(){
    const ctx2 = document.getElementById('myChart2') as HTMLCanvasElement;
    var labelGroups = [];
    var valueGroups = [];
    var backgroundGroup : any;

    for (let index = 0; index < this.content_sangres.length; index++) {
      for (let index2 = 0; index2 < this.content_diagnosticos.length; index2++) {
        if(this.content_diagnosticos[index2].id_sangre == this.content_sangres[index]._id){
          labelGroups.push(this.content_sangres[index].name);
        }
      }
    }

    labelGroups = labelGroups.filter((item, index, self) => self.indexOf(item) === index);

    for (let index = 0; index < labelGroups.length; index++) {
      valueGroups.push(0);
      for (let index2 = 0; index2 < this.content_diagnosticos.length; index2++) {
        if(this.sangreName(this.content_diagnosticos[index2].id_sangre) == labelGroups[index]){
          valueGroups[index] = valueGroups[index] + 1;
        }
      }
    }

    backgroundGroup = this.dynamicColors(labelGroups.length);

    const myChart2 = new Chart(ctx2, {
      type: 'pie', // Tipo de gráfico (puedes elegir otros como 'line', 'pie', etc.)
      data: {
        labels: labelGroups, // Etiquetas para el eje X
        datasets: [{
          label: 'Sangre',
          data: valueGroups, // Datos para el eje Y
          backgroundColor: backgroundGroup , // Color de fondo de las barras
          borderColor: backgroundGroup, // Color del borde de las barras
          borderWidth: 1, // Ancho del borde
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            position: 'right',
            display: true,
          },
          title: {
            display: false,
          }
        },
        maintainAspectRatio: false,
        elements: {
          point: {
            radius: 1
          }
        }
      }
    });
  }

  public getDonantesPorSangre(){
    const ctx3 = document.getElementById('myChart3') as HTMLCanvasElement;
    var labelGroups = [];
    var valueGroups = [];
    var backgroundGroup : any;

    for (let index = 0; index < this.content_sangres.length; index++) {
      for (let index2 = 0; index2 < this.content_usuarios.length; index2++) {
        if(this.content_usuarios[index2]._id == this.content_sangres[index].id_user){
          labelGroups.push(this.categoryName(this.content_sangres[index].grupo));
        }
      }
    }

    labelGroups = labelGroups.filter((item, index, self) => self.indexOf(item) === index);

    for (let index = 0; index < labelGroups.length; index++) {
      valueGroups.push(0);
      for (let index2 = 0; index2 < this.content_usuarios.length; index2++) {
        if(this.sangreGrupoByIdUser(this.content_usuarios[index2]._id) == labelGroups[index]){
          valueGroups[index] = valueGroups[index] + 1;
        }
      }
    }

    backgroundGroup = this.dynamicColors(labelGroups.length);

    const myChart2 = new Chart(ctx3, {
      type: 'line', // Tipo de gráfico (puedes elegir otros como 'line', 'pie', etc.)
      data: {
        labels: labelGroups, // Etiquetas para el eje X
        datasets: [{
          label: 'Donantes',
          data: valueGroups, // Datos para el eje Y
          backgroundColor: backgroundGroup , // Color de fondo de las barras
          borderColor: backgroundGroup, // Color del borde de las barras
          borderWidth: 1, // Ancho del borde
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            position: 'right',
            display: true,
          },
          title: {
            display: false,
          }
        },
        maintainAspectRatio: false,
        elements: {
          point: {
            radius: 1
          }
        }
      }
    });
  }

}


