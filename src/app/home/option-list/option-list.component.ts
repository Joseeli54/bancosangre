import { ChangeDetectorRef, Component, OnInit, ViewChild, ViewEncapsulation } from '@angular/core';
import { UserServices } from '../../services/userservices.services';
@Component({
  selector: 'app-option-list',
  templateUrl: './option-list.component.html',
  styleUrls: ['./option-list.component.css']
})
export class OptionListComponent implements OnInit{
  @ViewChild('searchInput', {static: true}) searchInput: any;

  constructor(public userService: UserServices) { 
    localStorage.setItem('view', '0');
  }
  productos : any 
  productosAll : any /*= [
    {
      title: "Estampados Vital", 
      content: "Se venden estampados de todo tipo para su tela.",
      price: "U$S10",
      status: "Disponible",
      category: "Hogar",
      image: "assets/img/imagen-bloque.jpg"
    },
    {
      title: "Perfumeria Las Rosas", 
      content: "Perfumeria creada por la sra Josefina Martinez",
      price: "U$S13",
      status: "Disponible",
      image: "assets/img/perfumes.jpg"
    },
    {
      title: "Estampados Vital", 
      content: "Se venden estampados de todo tipo para su tela.",
      price: "U$S33",
      status: "Disponible",
      image: "assets/img/perfumes-1.jpg"
    },
    {
      title: "Perfumeria Las Rosas", 
      content: "Perfumeria creada por la sra Josefina Martinez",
      price: "U$S54",
      status: "Disponible",
      image: "assets/img/fondo-armario.jpg"
    },
    {
      title: "Estampados Vital", 
      content: "Se venden estampados de todo tipo para su tela.",
      price: "U$S78",
      status: "Disponible",
      image: "assets/img/imagen-bloque.jpg"
    },
    {
      title: "Perfumeria Las Rosas", 
      content: "Perfumeria creada por la sra Josefina Martinez",
      price: "U$S92",
      status: "Agotado",
      image: "assets/img/perfumes.jpg"
    },
    {
      title: "Estampados Vitál", 
      content: "Se venden estampados de todo tipo para su tela.",
      price: "U$S23",
      status: "Disponible",
      image: "assets/img/perfumes-1.jpg"
    },
    {
      title: "Perfumeria Las Rosas", 
      content: "Perfumeria creada por la sra Josefina Martinez",
      price: "U$S23",
      status: "Disponible",
      image: "assets/img/fondo-armario.jpg"
    },
  ]*/
  public ngrid : number = 0;
  public bfilter : boolean = false;
  public bgridLayout : boolean = true;

  inputHTML=document.getElementById('searchInput');
  inputHTML2 =document.getElementById('searchInput2');

  ngOnInit() {
    //this.productos = this.productosAll;
    console.log("init");
    console.log(this.inputHTML);
    this.inputHTML?.addEventListener('keyup', (e) => {
      this.search();
    });

    this.inputHTML2?.addEventListener('keyup', (e) => {
      this.search();
    });

    this. hideGridLayout();
  }

  public refrescar(){
    location.reload();
  }

  public setGrid(num : number) : void{
    this.ngrid = num;
  }

  public setFilter() : void{
    if(this.bfilter){
      this.bfilter = false;
    }else{
      this.bfilter = true;
    }
  }

  public isEmptySearch() : boolean{
    let search = (document.getElementById('searchInput') as HTMLInputElement).value;
    if(search == "")
      search = (document.getElementById('searchInput2') as HTMLInputElement).value;

    if(search == ""){
      return true;
    }

    return false;
  }

  public getGridLayout(){
    if(this.ngrid == 0){
      return "col-md-3 col-12 my-md-2 my-2";
    } else if(this.ngrid == 1){
      return "col-md-4 col-12 my-md-2 my-2";
    }else{
      return "col-12 my-md-2 my-2";
    }
  }

  initProducts(){
    localStorage.setItem('spinner', '1');
    this.userService
      .getUrl('products')
      .then(response => {
        this.productos = response;
        this.productosAll = response;
        localStorage.setItem('spinner', '0');
      });
  }

  public hideGridLayout(){
    if(window.innerWidth <= 768){
      this.bgridLayout = false;
    }
  }

  search(){
    let search = (document.getElementById('searchInput') as HTMLInputElement).value;
    if(search == "")
      search = (document.getElementById('searchInput2') as HTMLInputElement).value;
    
    this.productos = this.productosAll.filter((producto: any) => {
      return this.eliminarDiacriticos(
        producto.title.toLowerCase() + 
        producto.content.toLowerCase() +
        producto.price.toString().toLowerCase() +
        producto.category.toLowerCase() +
        producto.currency.toLowerCase() +
        producto.status.toLowerCase()
      ).includes(this.eliminarDiacriticos(search.toLowerCase()));
    });
  }

  searchCategory(category: string){
    this.productos = this.productosAll.filter((producto: any) => {
      return this.eliminarDiacriticos(
        producto.category.toLowerCase()
      ).includes(this.eliminarDiacriticos(category.toLowerCase()));
    });
  }

  eliminarDiacriticos(texto: string) {
    return texto.normalize('NFD').replace(/[\u0300-\u036f]/g, '')
  }
 

}
