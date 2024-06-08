import { Component, ElementRef, Input, OnInit, ViewChild } from '@angular/core';
import { Table } from 'primeng/table';
import { Sangre } from 'src/app/models/sangre';
import { UserServices } from '../../services/userservices.services';
import { NgbModal, ModalDismissReasons } from '@ng-bootstrap/ng-bootstrap';
import { AlertService } from '../../services/alert/alert.service';
import { environment } from '../../../environments/environment';
import { User } from 'src/app/models/user';

@Component({
  selector: 'app-sangre',
  templateUrl: './sangre.component.html',
  styleUrls: ['./sangre.component.css']
})

export class SangreComponent {

  @Input() content_sangres : Array<Sangre> = [];
  
  @ViewChild('dt') dt? : Table;
  
  //Variables que serán utilizadas dentro de la tabla
  public _selected_columns: any = [];
  public lazy = false;
  public show_filters = true;

  //Objeto para guardar la información del sangreo seleccionado
  public sangre : Sangre = {};
  
  //Objetos que serán utilizado para llamar el modal
  public modalReference : any;
  public closeResult: string | undefined;
  public titleModal : string = "";
  public usuarios : Array<User> = [];

  public table_cols = [
    { field: 'code', header: 'Código' },
    { field: 'name', header: 'Nombre' },
    { field: 'level', header: 'Nivel de Sangre' },
    { field: 'grupo', header: 'Grupo Sanguineo' },
    { field: 'factor_rh', header: 'Factor RH' },
    { field: 'createBy', header: 'Creado Por' },
    { field: 'actions', header: 'Acciones' },
  ];

  public categories = [
    { code: "1", title: 'A' },
    { code: "2", title: 'AB' },
    { code: "3", title: 'B' },
    { code: "4", title: 'O' },
  ];

  public factores = [
    { code: "1", title: 'Positivo' },
    { code: "2", title: 'Negativo' },
  ];

  public tipo_donante = [
    { code: "1", title: 'Donantes voluntarios no remunerados' },
    { code: "2", title: 'Donantes a familiares o allegados' },
    { code: "3", title: 'Donantes voluntarios remunerados' },
    { code: "4", title: 'Donantes por aféresis' },
  ]

  private apiUrl : string = environment.apiUrl;
  public action = "";

  constructor(public userService: UserServices,
              private modalService: NgbModal,
              private alertService: AlertService) { 
    localStorage.setItem('view', '1');
  }

  ngOnInit(): void {
    this._selected_columns = this.table_cols;
    this.getSangre();
    this.getUsers();
  }

  public getSangre(){
    localStorage.setItem('spinner', '1');

    this.userService
      .getUrl('sangres')
      .then(response => {
        this.content_sangres = response;
        console.log(this.content_sangres)
        localStorage.setItem('spinner', '0');
      })
      .catch(error => {
        console.log(error.error)
        this.alertService.errorOcurred(error.statusText);
      });
  }

  public clearData(){
    this.sangre = new Sangre();
  }

  public deleteSangre(id : any){
    this.userService
    .deleteUrl('sangres/{id}', [id])
    .then(response => {
      console.log(response);
      this.getSangre();
      this.modalReference.close();
    })
    .catch(error => {
      console.log(error);
    })
  }

  openIA(content: any, data: any){
    if(data != null){
      this.sangre._id = data._id;
      this.sangre.code = data.code;
      this.sangre.name = data.name;
      this.sangre.content = data.content;
      this.sangre.level = data.level.replace("mm", "");
      this.sangre.type = data.type;
      this.sangre.createBy = data.createBy;
      this.sangre.grupo = data.grupo;
      this.sangre.factor_rh = data.factor_rh;
      this.sangre.genero = data.genero;
      this.sangre.globulos_rojos = data.globulos_rojos;
      this.sangre.hemoglobina = data.hemoglobina;
      this.sangre.hematocrito = data.hematocrito;
      this.sangre.globulos_blancos = data.globulos_blancos;
      this.sangre.plaquetas = data.plaquetas;
      this.sangre.date_donor = data.date_donor;
      this.sangre.date_due = data.date_due;
      this.sangre.id_user = data.id_user;
      this.titleModal = "Editar Sangre";

      this.action = "U";
    }else{
      this.sangre.code = this.content_sangres.length + 1;
      this.sangre.name = "Sangre #" + this.sangre.code;
      this.sangre.content = "";
      this.sangre.level = "75";
      this.sangre.type = "1";
      this.sangre.grupo = "1";
      this.sangre.factor_rh = "1";
      this.sangre.genero = "Hombre";
      this.sangre.globulos_rojos = "";
      this.sangre.hemoglobina = "";
      this.sangre.hematocrito = "";
      this.sangre.globulos_blancos = "";
      this.sangre.plaquetas = "";
      this.sangre.createBy = localStorage.getItem('username')!;
      this.titleModal = "Agregar Sangre";
      this.action = "C";
    }


    this.modalReference = this.modalService.open(content, { size: 'lg', centered: true});

    this.modalReference.result.then((result: any) => {
      this.closeResult = `Closed with: ${result}`;
    }, (reason: any) => {
      this.clearData();
      this.closeResult = `Dismissed ${this.getDismissReason(reason)}`;
    });    
  }

  open(content: any, data : any) {

    if(data != null){
      this.sangre._id = data._id;
      this.sangre.code = data.code;
      this.sangre.name = data.name;
      this.sangre.content = data.content;
      this.sangre.level = data.level.replace("mm", "");
      this.sangre.type = data.type;
      this.sangre.createBy = data.createBy;
      this.sangre.grupo = data.grupo;
      this.sangre.factor_rh = data.factor_rh;
      this.sangre.genero = data.genero;
      this.sangre.globulos_rojos = data.globulos_rojos;
      this.sangre.hemoglobina = data.hemoglobina;
      this.sangre.hematocrito = data.hematocrito;
      this.sangre.globulos_blancos = data.globulos_blancos;
      this.sangre.plaquetas = data.plaquetas;
      this.sangre.date_donor = data.date_donor;
      this.sangre.date_due = data.date_due;
      this.sangre.id_user = data.id_user;
      this.titleModal = "Editar Sangre";

      this.action = "U";
    }else{
      this.sangre.code = this.content_sangres.length + 1;
      this.sangre.name = "Sangre #" + this.sangre.code;
      this.sangre.content = "";
      this.sangre.level = "75";
      this.sangre.type = "1";
      this.sangre.grupo = "1";
      this.sangre.factor_rh = "1";
      this.sangre.genero = "Hombre";
      this.sangre.globulos_rojos = "";
      this.sangre.hemoglobina = "";
      this.sangre.hematocrito = "";
      this.sangre.globulos_blancos = "";
      this.sangre.plaquetas = "";
      this.sangre.createBy = localStorage.getItem('username')!;
      this.titleModal = "Agregar Sangre";
      this.action = "C";
    }

    this.modalReference = this.modalService.open(content, { size: 'lg', centered: true});

    this.modalReference.result.then((result: any) => {
      this.closeResult = `Closed with: ${result}`;
    }, (reason: any) => {
      this.clearData();
      this.closeResult = `Dismissed ${this.getDismissReason(reason)}`;
    });
  }

  private getDismissReason(reason: any): string {
    if (reason === ModalDismissReasons.ESC) {
      return 'by pressing ESC';
    } else if (reason === ModalDismissReasons.BACKDROP_CLICK) {
      return 'by clicking on a backdrop';
    } else {
      return  `with: ${reason}`;
    }
  }

  public getUsers(){
    this.userService.getUrl('users')
    .then(data => {

      this.usuarios = [];
      for (let index = 0; index < data.length; index++) {
        if(data[index].role == "USER"){
          this.usuarios.push(data[index]);
        }
      }
    })
    .catch(err => {
      console.log(err);
    })
  }

  public createOrUpdateSangre(){
    console.log(this.sangre);

    let sangres = {
      'code': this.sangre.code,
      'name': this.sangre.name,
      'content': this.sangre.content,
      'level': this.sangre.level! + "mm",
      'type': this.sangre.type,
      'grupo': this.sangre.grupo,
      'factor_rh': this.sangre.factor_rh,
      'genero': this.sangre.genero,
      'globulos_rojos': this.sangre.globulos_rojos,
      'hemoglobina': this.sangre.hemoglobina,
      'hematocrito': this.sangre.hematocrito,
      'globulos_blancos': this.sangre.globulos_blancos,
      'plaquetas': this.sangre.plaquetas,
      'date_donor': this.sangre.date_donor,
      'date_due': this.sangre.date_due,
      'id_user': this.sangre.id_user,
      'createBy': this.sangre.createBy }

    localStorage.setItem('spinner', '1');
    if(this.action == "C"){
      this.userService
        .postUrl('sangres', sangres)
        .then(response => {
          console.log(response)
          this.getSangre();
          this.modalReference.close();
        })
    }else{
      let id : any = this.sangre._id;
      this.userService
        .putUrl('sangres/{id}', sangres, [id])
        .then(response => {
          console.log(response);
          this.getSangre();
          this.modalReference.close();
        })
        .catch(error => {
          console.log(error);
        })
    }
  }

  public categoryName(code : any){
    let name = "No Aplicado";

    for (let index = 0; index < this.categories.length; index++) {
      if(this.categories[index].code == code){
        name = this.categories[index].title;
      }
    }

    return name;
  }

  public factorRHTitle(code : any){
    let name = "No Aplicado";

    for (let index = 0; index < this.factores.length; index++) {
      if(this.factores[index].code == code){
        name = this.factores[index].title;
      }
    }

    return name;
  }

  delete(content: any, data : any){
    if(data != null){
      this.sangre._id = data._id;
    }

    this.modalReference = this.modalService.open(content, { centered: true});

    this.modalReference.result.then((result: any) => {
      this.closeResult = `Closed with: ${result}`;
    }, (reason: any) => {
      this.clearData();
      this.closeResult = `Dismissed ${this.getDismissReason(reason)}`;
    });
  }

  @Input() get selected_columns(): any[] {
    return this._selected_columns;
  }

  set selected_columns(val: any[]) {
    //restore original order
    this._selected_columns = this.table_cols.filter(col => val.includes(col));
  }

  //PrimeNG Functions and Procedures
  apply_filter($event : any) {
    console.log($event.target.value);
    this.dt?.filterGlobal(($event.target as HTMLInputElement).value, 'contains');
  }

  clear(table: Table) {
    table.clear();
  }

  generarDiagnostico(sangre: any){
    console.log(sangre);
    this.userService
        .postUrl('diagnosticoAutomatico', 
        {
          name: 'Diagnostico automatico', 
          id_sangre: sangre._id,
          id_user: sangre.id_user,
          createBy: sangre.createBy,
        }
        )
        .then(response => {
          console.log(response)
          this.getSangre();
          this.modalReference.close();
        })
      
  }
}
